[English Version](./README.md)

# DPG 图像生成评估

基于 Lance 模型的 DPG 评估基准测试脚本。

## 文件说明

- `sample_DPG.py` - 推理 Python 脚本
- `sample_DPG.sh` - 启动脚本
- `DPG.jsonl` - 评估数据集

## 快速开始

### 基本用法

```bash
bash benchmarks/image_gen/DPG/sample_DPG.sh
```

运行前请直接修改 `benchmarks/image_gen/DPG/sample_DPG.sh` 顶部的“推理参数配置”区。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TASK_NAME` | `t2i` | 任务类型，DPG 固定为图像生成 |
| `VALIDATION_NUM_TIMESTEPS` | 50 | 推理步数 |
| `VALIDATION_TIMESTEP_SHIFT` | 3.5 | Timestep shift |
| `EVALUATION_SEED` | 42 | 随机种子 |
| `CFG_TEXT_SCALE` | 4.0 | CFG scale |
| `CFG_INTERVAL_START` | 0.4 | CFG 区间起点 |
| `CFG_INTERVAL_END` | 1.0 | CFG 区间终点 |
| `SAMPLE_NUM_PER_PROMPT` | 4 | 每个 case 生成的图像数量，用于拼接最终网格图 |
| `USE_KVCACHE` | `true` | 是否启用 KV cache |
| `NUM_GPUS` | 8 | GPU 数量 |
| `VIDEO_HEIGHT`/`VIDEO_WIDTH` | 768 | 图像分辨率 |
| `MODEL_PATH` | `downloads/Lance_3B` | Lance checkpoint 路径 |
| `VAL_DATASET_CONFIG_FILE` | `benchmarks/image_gen/DPG/DPG.jsonl` | 评估数据路径 |

## 修改方式

- 请手动编辑 `benchmarks/image_gen/DPG/sample_DPG.sh` 顶部的“推理参数配置”区。
- 修改完成后，直接运行 `bash benchmarks/image_gen/DPG/sample_DPG.sh`。
- `SAVE_PATH_GEN` 由脚本根据顶部参数自动生成，不需要手动设置。

## 保存格式

结果会按照以下结构保存：

```
results/DPG_ts50_tss3.5_seed42_cfg4.0_kvcache_20260507_120000/
├── 0.png
├── 1.png
├── 2.png
└── ...
```
