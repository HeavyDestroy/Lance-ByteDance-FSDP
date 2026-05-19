[English Version](./README.md)

# GenEVAL 图像生成评估

基于 Lance 模型的 GenEVAL 评估基准测试脚本。

## 文件说明

- `sample_GenEVAL.py` - 推理 Python 脚本
- `sample_GenEVAL.sh` - 启动脚本（推荐使用）
- `GenEVAL.jsonl` - 评估数据集

## 快速开始

### 基本用法

```bash
bash benchmarks/image_gen/GenEVAL/sample_GenEVAL.sh
```

运行前请直接修改 `benchmarks/image_gen/GenEVAL/sample_GenEVAL.sh` 顶部的“推理参数配置”区。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TASK_NAME` | `t2i` | 任务类型，GenEVAL 固定为图像生成 |
| `VALIDATION_NUM_TIMESTEPS` | 50 | 推理步数 |
| `VALIDATION_TIMESTEP_SHIFT` | 3.5 | Timestep shift |
| `EVALUATION_SEED` | 42 | 随机种子 |
| `CFG_TEXT_SCALE` | 4.0 | CFG scale |
| `CFG_INTERVAL_START` | 0.4 | CFG 区间起点 |
| `CFG_INTERVAL_END` | 1.0 | CFG 区间终点 |
| `SAMPLE_NUM_PER_PROMPT` | 4 | 每个 case 生成的图像数量（GenEVAL 默认为 4 张图） |
| `USE_KVCACHE` | `true` | 是否启用 KV cache |
| `NUM_GPUS` | 8 | GPU 数量 |
| `VIDEO_HEIGHT`/`VIDEO_WIDTH` | 768 | 图像分辨率 |
| `MODEL_PATH` | `downloads/Lance_3B` | Lance checkpoint 路径 |
| `VAL_DATASET_CONFIG_FILE` | `benchmarks/image_gen/GenEVAL/GenEVAL.jsonl` | 评估数据路径 |

## 修改方式

- 请手动编辑 `benchmarks/image_gen/GenEVAL/sample_GenEVAL.sh` 顶部的“推理参数配置”区。
- 修改完成后，直接运行 `bash benchmarks/image_gen/GenEVAL/sample_GenEVAL.sh`。
- `SAVE_PATH_GEN` 由脚本根据顶部参数自动生成，不需要手动设置。

## 保存格式

结果会按照以下结构保存：

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

每个案例生成 4 张图像（`sample_num_per_prompt=4`）。

## 注意事项

- 如果需要切换模型、数据集或分辨率，请直接修改脚本顶部配置。
- ViT 路径默认由代码内部自动解析，无需单独配置。
