[English Version](./README.md)

# GEdit 图像编辑评估

基于 Lance 模型的 GEdit 评估基准测试脚本。

## 文件说明

- `sample_GEdit.py` - 推理 Python 脚本
- `sample_GEdit.sh` - 启动脚本
- `GEdit_en.json` - 评估数据集

## 快速开始

### 基本用法

```bash
bash benchmarks/image_gen/GEdit/sample_GEdit.sh
```

运行前请直接修改 `benchmarks/image_gen/GEdit/sample_GEdit.sh` 顶部的“推理参数配置”区。
请参考 `https://github.com/stepfun-ai/Step1X-Edit` 下载 GEdit-Bench 的源图，并将所有图片放到 `benchmarks/image_gen/GEdit/images/` 中。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TASK_NAME` | `image_edit` | 任务类型，GEdit 固定为图像编辑 |
| `VALIDATION_NUM_TIMESTEPS` | 50 | 推理步数 |
| `VALIDATION_TIMESTEP_SHIFT` | 3.5 | Timestep shift |
| `EVALUATION_SEED` | 42 | 随机种子 |
| `CFG_TEXT_SCALE` | 4.0 | CFG scale |
| `CFG_INTERVAL_START` | 0.4 | CFG 区间起点 |
| `CFG_INTERVAL_END` | 1.0 | CFG 区间终点 |
| `USE_KVCACHE` | `true` | 是否启用 KV cache |
| `NUM_GPUS` | 8 | GPU 数量 |
| `MODEL_PATH` | `downloads/Lance_3B` | Lance checkpoint 路径 |
| `VAL_DATASET_CONFIG_FILE` | `benchmarks/image_gen/GEdit/GEdit_en.json` | 评估数据路径 |

## 修改方式

- 请手动编辑 `benchmarks/image_gen/GEdit/sample_GEdit.sh` 顶部的“推理参数配置”区。
- 修改完成后，直接运行 `bash benchmarks/image_gen/GEdit/sample_GEdit.sh`。
- `SAVE_PATH_GEN` 由脚本根据顶部参数自动生成，不需要手动设置。

## 保存格式

结果会按照以下结构保存：

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

每个 case 默认生成 1 张编辑结果图，并按 `task_type/instruction_language/key` 分目录保存为 `.webp` 文件；同时会额外写出 `prompt.json` 用于记录生成文本。
## 注意事项

- 如果需要切换模型、数据集或分辨率，请直接修改脚本顶部配置。
- 默认结果目录会自动包含关键参数和时间戳，方便区分不同实验。
