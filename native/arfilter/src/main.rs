//! arfilter — AudioRefine streaming speech filter.
//!
//! Reads mono 32-bit float little-endian PCM from stdin and writes exactly the
//! same number of samples to stdout, time-aligned with the input (all algorithmic
//! delay is compensated). Processing chain, every stage optional:
//!
//!   high-pass (Butterworth) -> mains-hum notches -> DeepFilterNet3 -> peaking EQ
//!
//! DeepFilterNet3 (MIT OR Apache-2.0, Hendrik Schröter) runs through its Rust/tract
//! implementation with an attenuation limit, i.e. the enhanced spectrum is mixed
//! with the unprocessed spectrum so that noise (and anything the network mistakes
//! for noise) is reduced by at most `--df-atten` dB. Nothing is ever gated to silence.

use std::f64::consts::PI;
use std::io::{self, BufWriter, Read, Write};
use std::process::exit;

use df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::prelude::*;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    fn new(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Biquad { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0, z1: 0., z2: 0. }
    }
    /// RBJ cookbook high-pass.
    fn highpass(sr: f64, f0: f64, q: f64) -> Self {
        let w0 = 2. * PI * f0 / sr;
        let (s, c) = w0.sin_cos();
        let alpha = s / (2. * q);
        Biquad::new((1. + c) / 2., -(1. + c), (1. + c) / 2., 1. + alpha, -2. * c, 1. - alpha)
    }
    /// RBJ cookbook notch.
    fn notch(sr: f64, f0: f64, q: f64) -> Self {
        let w0 = 2. * PI * f0 / sr;
        let (s, c) = w0.sin_cos();
        let alpha = s / (2. * q);
        Biquad::new(1., -2. * c, 1., 1. + alpha, -2. * c, 1. - alpha)
    }
    /// RBJ cookbook peaking EQ.
    fn peaking(sr: f64, f0: f64, gain_db: f64, q: f64) -> Self {
        let a = 10f64.powf(gain_db / 40.);
        let w0 = 2. * PI * f0 / sr;
        let (s, c) = w0.sin_cos();
        let alpha = s / (2. * q);
        Biquad::new(1. + alpha * a, -2. * c, 1. - alpha * a, 1. + alpha / a, -2. * c, 1. - alpha / a)
    }
    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        // Transposed direct form II
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

struct Chain(Vec<Biquad>);

impl Chain {
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let mut v = x as f64;
        for bq in self.0.iter_mut() {
            v = bq.process(v);
        }
        v as f32
    }
    fn process_slice(&mut self, buf: &mut [f32]) {
        if self.0.is_empty() {
            return;
        }
        for s in buf.iter_mut() {
            *s = self.process(*s);
        }
    }
}

struct Args {
    sr: f64,
    hpf: f64,
    hpf_order: u32,
    notch: f64,
    notch_count: u32,
    notch_q: f64,
    df_atten: f32,
    peaks: Vec<(f64, f64, f64)>,
    lsnr_out: Option<String>,
    info: bool,
}

fn usage() -> ! {
    eprintln!(
        "arfilter {VERSION}\n\
         usage: arfilter --sr 48000 [--hpf HZ] [--hpf-order 2|4] [--notch HZ --notch-count N --notch-q Q]\n\
         \x20               [--df-atten DB] [--peak F:GAIN_DB:Q]... [--lsnr-out PATH] [--info]\n\
         stdin/stdout: mono f32le PCM; output has exactly the input length and no delay."
    );
    exit(2)
}

fn parse_args() -> Args {
    let mut a = Args {
        sr: 0.,
        hpf: 0.,
        hpf_order: 2,
        notch: 0.,
        notch_count: 0,
        notch_q: 30.,
        df_atten: 0.,
        peaks: vec![],
        lsnr_out: None,
        info: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let num = |v: Option<&String>| -> f64 {
        v.and_then(|s| s.parse::<f64>().ok()).unwrap_or_else(|| usage())
    };
    while i < argv.len() {
        let k = argv[i].as_str();
        let v = argv.get(i + 1);
        match k {
            "--sr" => a.sr = num(v),
            "--hpf" => a.hpf = num(v),
            "--hpf-order" => a.hpf_order = num(v) as u32,
            "--notch" => a.notch = num(v),
            "--notch-count" => a.notch_count = num(v) as u32,
            "--notch-q" => a.notch_q = num(v),
            "--df-atten" => a.df_atten = num(v) as f32,
            "--peak" => {
                let s = v.unwrap_or_else(|| usage());
                let p: Vec<f64> = s.split(':').filter_map(|x| x.parse().ok()).collect();
                if p.len() != 3 {
                    usage()
                }
                a.peaks.push((p[0], p[1], p[2]));
            }
            "--lsnr-out" => a.lsnr_out = Some(v.unwrap_or_else(|| usage()).clone()),
            "--info" => {
                a.info = true;
                i += 1;
                continue;
            }
            "--version" => {
                println!("arfilter {VERSION}");
                exit(0)
            }
            _ => usage(),
        }
        i += 2;
    }
    if a.sr <= 0. && !a.info {
        usage()
    }
    a
}

/// Fill `buf` from `r`; returns number of complete f32 samples read (short only at EOF).
fn read_samples(r: &mut impl Read, bytes: &mut [u8], buf: &mut [f32]) -> io::Result<usize> {
    let want = buf.len() * 4;
    let mut got = 0;
    while got < want {
        let n = r.read(&mut bytes[got..want])?;
        if n == 0 {
            break;
        }
        got += n;
    }
    if got % 4 != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "input is not whole f32 samples"));
    }
    let n = got / 4;
    for (i, s) in buf[..n].iter_mut().enumerate() {
        *s = f32::from_le_bytes([bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]]);
    }
    Ok(n)
}

fn write_samples(w: &mut impl Write, buf: &[f32]) -> io::Result<()> {
    let mut out = Vec::with_capacity(buf.len() * 4);
    for s in buf {
        out.extend_from_slice(&s.to_le_bytes());
    }
    w.write_all(&out)
}

/// Deterministic xorshift for the low-level guard noise.
struct XorShift(u64);
impl XorShift {
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

fn run(a: Args) -> anyhow_like::Result<()> {
    let sr = a.sr;
    let mut pre = Chain(vec![]);
    if a.hpf > 0. {
        if a.hpf_order >= 4 {
            // 4th-order Butterworth = two biquads
            pre.0.push(Biquad::highpass(sr, a.hpf, 0.541_196_100_146_197));
            pre.0.push(Biquad::highpass(sr, a.hpf, 1.306_562_964_876_376_5));
        } else {
            pre.0.push(Biquad::highpass(sr, a.hpf, std::f64::consts::FRAC_1_SQRT_2));
        }
    }
    if a.notch > 0. {
        for h in 1..=a.notch_count.max(1) {
            let f = a.notch * h as f64;
            if f < sr * 0.45 {
                pre.0.push(Biquad::notch(sr, f, a.notch_q));
            }
        }
    }
    let mut post = Chain(a.peaks.iter().map(|&(f, g, q)| Biquad::peaking(sr, f, g, q)).collect());

    let stdin = io::stdin();
    let mut rd = stdin.lock();
    let stdout = io::stdout();
    let mut wr = BufWriter::with_capacity(1 << 20, stdout.lock());

    if a.df_atten <= 0. {
        // Pure IIR chain: sample-synchronous, zero added latency.
        const BLK: usize = 65536;
        let mut bytes = vec![0u8; BLK * 4];
        let mut buf = vec![0f32; BLK];
        loop {
            let n = read_samples(&mut rd, &mut bytes, &mut buf)?;
            if n == 0 {
                break;
            }
            pre.process_slice(&mut buf[..n]);
            post.process_slice(&mut buf[..n]);
            write_samples(&mut wr, &buf[..n])?;
            if n < BLK {
                break;
            }
        }
        wr.flush()?;
        return Ok(());
    }

    if (sr - 48000.).abs() > 0.5 {
        return Err("DeepFilterNet requires --sr 48000".into());
    }
    let r_params = RuntimeParams::default_with_ch(1)
        .with_atten_lim(a.df_atten)
        .with_thresholds(-15., 35., 35.);
    let df_params = DfParams::default();
    let mut model = DfTract::new(df_params, &r_params).map_err(|e| format!("DeepFilterNet init failed: {e}"))?;
    let hop = model.hop_size;
    let delay = model.fft_size - model.hop_size + model.lookahead * model.hop_size;

    let mut lsnr_w = match &a.lsnr_out {
        Some(p) => Some(BufWriter::new(std::fs::File::create(p)?)),
        None => None,
    };

    let mut bytes = vec![0u8; hop * 4];
    let mut inp = vec![0f32; hop];
    let mut noisy = Array2::<f32>::zeros((1, hop));
    let mut enh = Array2::<f32>::zeros((1, hop));
    let mut total_in: usize = 0;
    let mut total_out: usize = 0;
    let mut to_skip = delay;
    let mut eof = false;
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let guard_rms: f32 = 4e-4; // about -68 dBFS, only used for near-digital-silence frames

    loop {
        let n = if eof { 0 } else { read_samples(&mut rd, &mut bytes, &mut inp)? };
        if n < hop {
            eof = true;
        }
        total_in += n;
        for s in inp[n..].iter_mut() {
            *s = 0.;
        }
        pre.process_slice(&mut inp[..n]);
        // DeepFilterNet zeroes frames whose mean square is below 1e-7 and skips its state update.
        // To avoid such artificial silence/glitches, near-silent frames get a tiny deterministic
        // noise floor, which the network then attenuates like any other noise.
        let ms: f32 = inp.iter().map(|x| x * x).sum::<f32>() / hop as f32;
        let guard = ms < 2e-7;
        for (j, s) in noisy.row_mut(0).iter_mut().enumerate() {
            let mut v = inp[j];
            if guard {
                v += rng.next_f32() * guard_rms * 3.4641; // uniform with the given RMS
            }
            *s = v;
        }
        let lsnr = model
            .process(noisy.view(), enh.view_mut())
            .map_err(|e| format!("DeepFilterNet processing failed: {e}"))?;
        if let Some(w) = lsnr_w.as_mut() {
            w.write_all(&lsnr.to_le_bytes())?;
        }
        let mut frame: Vec<f32> = enh.row(0).to_vec();
        let mut start = 0;
        if to_skip > 0 {
            start = to_skip.min(hop);
            to_skip -= start;
        }
        let remaining = total_in.saturating_sub(total_out);
        let take = (hop - start).min(remaining);
        if take > 0 {
            let out = &mut frame[start..start + take];
            post.process_slice(out);
            write_samples(&mut wr, out)?;
            total_out += take;
        }
        if eof && total_out >= total_in {
            break;
        }
    }
    wr.flush()?;
    if let Some(mut w) = lsnr_w {
        w.flush()?;
    }
    Ok(())
}

mod anyhow_like {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
}

fn main() {
    let a = parse_args();
    if a.info {
        let r = RuntimeParams::default_with_ch(1);
        match DfTract::new(DfParams::default(), &r) {
            Ok(m) => {
                println!(
                    "{{\"version\":\"{VERSION}\",\"df_sr\":{},\"hop\":{},\"fft\":{},\"lookahead\":{},\"delay\":{}}}",
                    m.sr,
                    m.hop_size,
                    m.fft_size,
                    m.lookahead,
                    m.fft_size - m.hop_size + m.lookahead * m.hop_size
                );
                exit(0)
            }
            Err(e) => {
                eprintln!("DeepFilterNet init failed: {e}");
                exit(1)
            }
        }
    }
    if let Err(e) = run(a) {
        // A closed stdout (consumer cancelled) is not an error worth a stack of noise.
        eprintln!("arfilter error: {e}");
        exit(1)
    }
}
