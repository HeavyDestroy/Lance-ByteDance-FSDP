[Chinese Version](./README_zh.md)

# GEdit Image Editing Evaluation

Benchmark evaluation scripts for GEdit based on the Lance model.

## Files

- `sample_GEdit.py` - Python inference script
- `sample_GEdit.sh` - Launch script
- `GEdit_en.json` - Evaluation dataset

## Quick Start

### Basic Usage

```bash
bash benchmarks/image_gen/GEdit/sample_GEdit.sh
```

Before running, edit the "Inference Parameters" section at the top of `benchmarks/image_gen/GEdit/sample_GEdit.sh`.
Please follow `https://github.com/stepfun-ai/Step1X-Edit` to download the source images in GEdit-Bench and put all images in `benchmarks/image_gen/GEdit/images/`.

## Parameters

| Parameter | Default | Description |
|------|--------|------|
| `TASK_NAME` | `image_edit` | Task type. GEdit is fixed to image editing. |
| `VALIDATION_NUM_TIMESTEPS` | 50 | Number of inference steps. |
| `VALIDATION_TIMESTEP_SHIFT` | 3.5 | Timestep shift. |
| `EVALUATION_SEED` | 42 | Random seed. |
| `CFG_TEXT_SCALE` | 4.0 | CFG scale. |
| `CFG_INTERVAL_START` | 0.4 | Start of the CFG interval. |
| `CFG_INTERVAL_END` | 1.0 | End of the CFG interval. |
| `USE_KVCACHE` | `true` | Whether to enable KV cache. |
| `NUM_GPUS` | 8 | Number of GPUs. |
| `MODEL_PATH` | `downloads/Lance_3B` | Path to the Lance checkpoint. |
| `VAL_DATASET_CONFIG_FILE` | `benchmarks/image_gen/GEdit/GEdit_en.json` | Path to the evaluation data. |

## How To Modify

- Edit the "Inference Parameters" section at the top of `benchmarks/image_gen/GEdit/sample_GEdit.sh`.
- After updating the parameters, run `bash benchmarks/image_gen/GEdit/sample_GEdit.sh` directly.
- `SAVE_PATH_GEN` is generated automatically from the script parameters and does not need to be set manually.

## Output Format

Results are saved in a structure like this:

```
results/GEdit_ts50_tss3.5_seed42_cfg4.0_kvcache_20260507_120000/
├── fullset/
│   ├── add/
│   │   ├── en/
│   │   │   ├── 000001.webp
│   │   │   └── ...
│   ├── remove/
│   │   └── en/
│   │       └── ...
├── prompt.json
```

Each case generates one edited image by default and stores it as a `.webp` file under `task_type/instruction_language/key`. A `prompt.json` file is also written to record the generated text.

## Notes

- If you need to switch the model, dataset, or resolution, edit the script configuration at the top directly.
- The default result directory automatically includes key parameters and a timestamp for easier experiment tracking.
