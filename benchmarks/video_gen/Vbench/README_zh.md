[English Version](./README.md)

# VBench 视频生成评估

基于 Lance 模型的 VBench 评估基准测试脚本。

## 文件说明

- `sample_vbench.py` - 推理 Python 脚本
- `sample_vbench.sh` - 启动脚本（推荐使用）
- `Vbench_recaption.jsonl` - 评估数据集

## 快速开始

### 基本用法

```bash
bash sample_vbench.sh
```

运行前请直接修改 `benchmarks/video_gen/Vbench/sample_vbench.sh` 顶部的“推理参数配置”区。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TASK_NAME` | `t2v` | 任务类型，VBench 固定为视频生成 |
| `VALIDATION_NUM_TIMESTEPS` | 50 | 推理步数 |
| `VALIDATION_TIMESTEP_SHIFT` | 3.5 | Timestep shift |
| `EVALUATION_SEED` | 42 | 随机种子 |
| `CFG_TEXT_SCALE` | 4.0 | CFG scale |
| `CFG_INTERVAL_START` | 0.4 | CFG 区间起点 |
| `CFG_INTERVAL_END` | 1.0 | CFG 区间终点 |
| `SAMPLE_NUM_PER_PROMPT` | 5 | 每个普通 prompt 生成的视频数量 |
| `USE_KVCACHE` | `true` | 是否启用 KV cache |
| `NUM_GPUS` | 8 | GPU 数量 |
| `VIDEO_HEIGHT`/`VIDEO_WIDTH` | 480 | 视频分辨率 |
| `NUM_FRAMES` | 50 | 输出视频帧数 |
| `MAX_NUM_FRAMES` | 121 | 单个样本最大帧数 |
| `MAX_LATENT_SIZE` | 64 | latent size 上限 |
| `RESOLUTION` | `video_480p` | 数据集分辨率标签 |
| `MODEL_PATH` | `downloads/Lance_3B_Video` | Lance checkpoint 路径 |
| `VAL_DATASET_CONFIG_FILE` | `benchmarks/video_gen/Vbench/Vbench_recaption.jsonl` | 评估数据路径 |
| `CONFIG_JSON_PATH` | `""` | 可选训练配置 JSON |

## 修改方式

- 请手动编辑 `benchmarks/video_gen/Vbench/sample_vbench.sh` 顶部的“推理参数配置”区。
- 修改完成后，直接运行 `bash benchmarks/video_gen/Vbench/sample_vbench.sh`。
- `SAVE_PATH_GEN` 由脚本根据顶部参数自动生成，不需要手动设置。

## 保存格式

结果会按照以下结构保存：

```
results/Vbench_ts50_tss3.5_seed42_cfg4.0_kvcache_20260507_120000/
├── In a still frame, a stop sign-0.mp4
├── In a still frame, a stop sign-1.mp4
├── a toilet, frozen in time-0.mp4
├── ...
├── prompt.json
```

每个 prompt 默认生成 `SAMPLE_NUM_PER_PROMPT` 个视频，并按 `原始 prompt-采样序号.mp4` 命名；同时会额外写出 `prompt.json` 记录生成文本。
如果仓库中存在 `temporal_flickering_prompts.json`，对应 prompt 会自动提升采样数；当前文件不存在时，脚本会直接使用 `SAMPLE_NUM_PER_PROMPT`。

## 注意事项

- 如果需要切换模型、数据集、帧数或分辨率，请直接修改脚本顶部配置。
- ViT 路径默认由代码内部自动解析，无需单独配置。
- `CONFIG_JSON_PATH` 仅作为可选训练配置 JSON 传入，不会替代脚本顶部其它显式参数。
