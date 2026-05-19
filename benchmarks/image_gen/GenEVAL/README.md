[Chinese Version](./README_zh.md)

# GenEVAL Image Generation Evaluation

Benchmark evaluation scripts for GenEVAL based on the Lance model.

## Files

- `sample_GenEVAL.py` - Python inference script
- `sample_GenEVAL.sh` - Launch script (recommended)
- `GenEVAL.jsonl` - Evaluation dataset

## Quick Start

### Basic Usage

```bash
bash benchmarks/image_gen/GenEVAL/sample_GenEVAL.sh
```

Before running, edit the "Inference Parameters" section at the top of `benchmarks/image_gen/GenEVAL/sample_GenEVAL.sh`.

## Parameters

| Parameter | Default | Description |
|------|--------|------|
| `TASK_NAME` | `t2i` | Task type. GenEVAL is fixed to image generation. |
| `VALIDATION_NUM_TIMESTEPS` | 50 | Number of inference steps. |
| `VALIDATION_TIMESTEP_SHIFT` | 3.5 | Timestep shift. |
| `EVALUATION_SEED` | 42 | Random seed. |
| `CFG_TEXT_SCALE` | 4.0 | CFG scale. |
| `CFG_INTERVAL_START` | 0.4 | Start of the CFG interval. |
| `CFG_INTERVAL_END` | 1.0 | End of the CFG interval. |
| `SAMPLE_NUM_PER_PROMPT` | 4 | Number of images generated per case. GenEVAL defaults to 4 images. |
| `USE_KVCACHE` | `true` | Whether to enable KV cache. |
| `NUM_GPUS` | 8 | Number of GPUs. |
| `VIDEO_HEIGHT`/`VIDEO_WIDTH` | 768 | Image resolution. |
| `MODEL_PATH` | `downloads/Lance_3B` | Path to the Lance checkpoint. |
| `VAL_DATASET_CONFIG_FILE` | `benchmarks/image_gen/GenEVAL/GenEVAL.jsonl` | Path to the evaluation data. |

## How To Modify

- Edit the "Inference Parameters" section at the top of `benchmarks/image_gen/GenEVAL/sample_GenEVAL.sh`.
- After updating the parameters, run `bash benchmarks/image_gen/GenEVAL/sample_GenEVAL.sh` directly.
- `SAVE_PATH_GEN` is generated automatically from the script parameters and does not need to be set manually.

## Output Format

Results are saved in a structure like this:

```
results/GenEVAL_ts50_tss3.5_seed42_cfg4.0_kvcache_20260507_120000/
├── 00000/
│   ├── metadata.jsonl
│   ├── grid.png
│   └── samples/
│       ├── 0.png
│       ├── 1.png
│       ├── 2.png
│       └── 3.png
├── 00001/
│   ├── metadata.jsonl
│   ├── grid.png
│   └── samples/
│       ...
```

Each case generates 4 images by default (`sample_num_per_prompt=4`).

## Notes

- If you need to switch the model, dataset, or resolution, edit the script configuration at the top directly.
- The ViT path is resolved automatically by the code and usually does not need to be configured separately.
