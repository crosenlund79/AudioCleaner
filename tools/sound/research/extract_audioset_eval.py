"""Stream the AudioSet eval split from the HF mirror, convert every clip to 32 kHz mono FLAC (16-bit)."""
import io, json, os, sys, time, concurrent.futures as cf
import numpy as np, soundfile as sf, pyarrow.parquet as pq
from scipy.signal import resample_poly
from huggingface_hub import hf_hub_download

OUT = "/home/user/sed/as_eval"; os.makedirs(OUT, exist_ok=True)
META = os.path.join(OUT, "labels.jsonl")
done = set()
if os.path.exists(META):
    done = {json.loads(l)["shard"] for l in open(META) if '"shard"' in l}

def convert(row):
    vid, audio, labels = row
    try:
        x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    except Exception as e:
        return vid, None, str(e)
    x = x.mean(1)
    g = np.gcd(32000, sr)
    y = resample_poly(x, 32000 // g, sr // g).astype(np.float32)
    sf.write(os.path.join(OUT, vid + ".flac"), np.clip(y, -1, 1), 32000, subtype="PCM_16")
    return vid, len(y) / 32000, labels

def shard(i):
    name = f"{i:02d}"
    p = hf_hub_download("agkphysics/AudioSet", f"data/eval/{name}.parquet", repo_type="dataset",
                        local_dir="/home/user/sed/tmp_parquet")
    pf = pq.ParquetFile(p)
    out = []
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=["video_id", "audio", "labels"]).to_pylist()
        with cf.ThreadPoolExecutor(4) as ex:
            out += list(ex.map(convert, [(r["video_id"], r["audio"], r["labels"]) for r in t]))
    os.remove(p)
    return name, out

todo = [i for i in range(35) if f"{i:02d}" not in done]
t0 = time.time()
with cf.ThreadPoolExecutor(2) as ex, open(META, "a") as f:
    for name, out in ex.map(shard, todo):
        for vid, dur, lab in out:
            f.write(json.dumps({"vid": vid, "dur": dur, "labels": lab if dur else None, "err": None if dur else lab}) + "\n")
        f.write(json.dumps({"shard": name}) + "\n"); f.flush()
        print(f"shard {name} done, {len(out)} clips, {time.time()-t0:.0f}s", flush=True)
print("ALL DONE", flush=True)
