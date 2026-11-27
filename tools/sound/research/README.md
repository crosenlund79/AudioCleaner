# Research scripts of the sound-analysis evaluation

Scripts used for the model comparison and data preparation in ACOUSTIC_ANALYSIS.md. They ran in a
separate evaluation workspace (`/home/user/sed`) with PyTorch, the EfficientAT repository, PANNs
(`panns-inference`), Hugging Face `transformers` and the downloaded datasets; paths are those of
that workspace. They are kept for review and reproduction, not used by the application.

| Script | Purpose |
|---|---|
| `extract_audioset_eval.py` | stream the AudioSet evaluation split (HF mirror `agkphysics/AudioSet`) to 32 kHz FLAC |
| `vocab_candidates.py` | first version of the candidate categories (AudioSet class indices) |
| `score2.py` | EfficientAT models: 10 s clip, per-time-step head (saturates), padded 2 s windows |
| `score_models.py` | EfficientAT / PANNs CNN14 / CNN14-DecisionLevelMax / AST scoring |
| `cmp_table.py` | per-category AP table of the comparison (§7.1) |
| `bench.py` | time per 10 s clip of each candidate (4 threads) |
| `make_backgrounds.py` | event-free background material (FLEURS read speech, AMI meetings, quiet room) |
| `frontend_np.py` | first numpy version of the model front-end (verified against PyTorch) |

The application-side tools are one level up: `score_clips.py`, `score_context.py`,
`score_long.py`, `tune.py`, `make_mixtures.py`, `eval_mix.py`, `report_tables.py`,
`export_onnx.py`.
