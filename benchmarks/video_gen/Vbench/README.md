[Chinese Version](./README_zh.md)

# VBench Video Generation Evaluation

Benchmark evaluation scripts for VBench based on the Lance model.

## Files

- `sample_vbench.py` - Python inference script
- `sample_vbench.sh` - Launch script (recommended)
- `Vbench_recaption.jsonl` - Evaluation dataset

## Quick Start

### Basic Usage

```bash
bash sample_vbench.sh
```

Before running, edit the "Inference Parameters" section at the top of `benchmarks/video_gen/Vbench/sample_vbench.sh`.

## Parameters

| Parameter | Default | Description |
|------|--------|------|
| `TASK_NAME` | `t2v` | Task type. VBench is fixed to video generation. |
| `VALIDATION_NUM_TIMESTEPS` | 50 | Number of inference steps. |
| `VALIDATION_TIMESTEP_SHIFT` | 3.5 | Timestep shift. |
| `EVALUATION_SEED` | 42 | Random seed. |
| `CFG_TEXT_SCALE` | 4.0 | CFG scale. |
| `CFG_INTERVAL_START` | 0.4 | Start of the CFG interval. |
| `CFG_INTERVAL_END` | 1.0 | End of the CFG interval. |
| `SAMPLE_NUM_PER_PROMPT` | 5 | Number of videos generated for each regular prompt. |
| `USE_KVCACHE` | `true` | Whether to enable KV cache. |
| `NUM_GPUS` | 8 | Number of GPUs. |
| `VIDEO_HEIGHT`/`VIDEO_WIDTH` | 480 | Video resolution. |
| `NUM_FRAMES` | 50 | Number of output video frames. |
| `MAX_NUM_FRAMES` | 121 | Maximum number of frames per sample. |
| `MAX_LATENT_SIZE` | 64 | Maximum latent size. |
| `RESOLUTION` | `video_480p` | Dataset resolution tag. |
| `MODEL_PATH` | `downloads/Lance_3B_Video` | Path to the Lance checkpoint. |
| `VAL_DATASET_CONFIG_FILE` | `benchmarks/video_gen/Vbench/Vbench_recaption.jsonl` | Path to the evaluation data. |
| `CONFIG_JSON_PATH` | `""` | Optional training configuration JSON. |

## How To Modify

- Edit the "Inference Parameters" section at the top of `benchmarks/video_gen/Vbench/sample_vbench.sh`.
- After updating the parameters, run `bash benchmarks/video_gen/Vbench/sample_vbench.sh` directly.
- `SAVE_PATH_GEN` is generated automatically from the script parameters and does not need to be set manually.

## Output Format

Results are saved in a structure like this:

```
results/Vbench_ts50_tss3.5_seed42_cfg4.0_kvcache_20260507_120000/
├── In a still frame, a stop sign-0.mp4
├── In a still frame, a stop sign-1.mp4
├── a toilet, frozen in time-0.mp4
├── ...
├── prompt.json
```

Each prompt generates `SAMPLE_NUM_PER_PROMPT` videos by default, named as `original-prompt-sample-index.mp4`. A `prompt.json` file is also written to record the generated text.
If `temporal_flickering_prompts.json` exists in the repository, the corresponding prompts automatically use a larger sample count. If the file does not exist, the script directly uses `SAMPLE_NUM_PER_PROMPT`.

## Notes

- If you need to switch the model, dataset, frame count, or resolution, edit the script configuration at the top directly.
- The ViT path is resolved automatically by the code and usually does not need to be configured separately.
- `CONFIG_JSON_PATH` is only passed through as an optional training configuration JSON and does not override the other explicit script parameters.
