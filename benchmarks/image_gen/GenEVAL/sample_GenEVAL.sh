#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../sample_env.sh"

# ========================= 推理参数配置 =========================
TASK_NAME="t2i"
NUM_GPUS=8

VALIDATION_NUM_TIMESTEPS=50
VALIDATION_TIMESTEP_SHIFT=3.5
EVALUATION_SEED=42
CFG_TEXT_SCALE=4.0
CFG_INTERVAL_START=0.4
CFG_INTERVAL_END=1.0
SAMPLE_NUM_PER_PROMPT=4
USE_KVCACHE=true

VIDEO_HEIGHT=768
VIDEO_WIDTH=768

MODEL_PATH="downloads/Lance_3B"
VAL_DATASET_CONFIG_FILE="benchmarks/image_gen/GenEVAL/GenEVAL.jsonl"

# ========================= 自动生成路径 =========================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
KVCACHE_TAG=""
if [ "$USE_KVCACHE" = "true" ]; then
    KVCACHE_TAG="kvcache_"
fi
SAVE_PATH_GEN="results/GenEVAL_ts${VALIDATION_NUM_TIMESTEPS}_tss${VALIDATION_TIMESTEP_SHIFT}_seed${EVALUATION_SEED}_cfg${CFG_TEXT_SCALE}_${KVCACHE_TAG}${TIMESTAMP}"

if [ -z "$MODEL_PATH" ]; then
    echo "错误: 请在脚本顶部配置区手动设置 MODEL_PATH"
    exit 1
fi

# ============================== 环境与分布式配置 ==============================
lance_setup_common_env
lance_setup_distributed_env "$NUM_GPUS"
lance_setup_shard_env 1

# ========================= 显示任务配置 =========================
echo "================================================"
echo "GenEVAL T2I 推理"
echo "================================================"
echo "GPU数量: ${NUM_GPUS}"
echo "保存路径: ${SAVE_PATH_GEN}"
echo "分辨率: ${VIDEO_HEIGHT}x${VIDEO_WIDTH}"
echo "模型路径: ${MODEL_PATH}"
if [ -n "$VAL_DATASET_CONFIG_FILE" ]; then
    echo "数据路径: ${VAL_DATASET_CONFIG_FILE}"
fi
echo ""
echo "关键参数："
echo "  - validation_num_timesteps: ${VALIDATION_NUM_TIMESTEPS}"
echo "  - validation_timestep_shift: ${VALIDATION_TIMESTEP_SHIFT}"
echo "  - evaluation_seed: ${EVALUATION_SEED}"
echo "  - cfg_text_scale: ${CFG_TEXT_SCALE}"
echo "  - cfg_interval: [${CFG_INTERVAL_START}, ${CFG_INTERVAL_END}]"
echo "  - sample_num_per_prompt: ${SAMPLE_NUM_PER_PROMPT}"
echo "  - use_KVcache: ${USE_KVCACHE}"
echo "================================================"
echo ""

# ============================== 执行推理 ==============================
# 注意：请直接修改本脚本顶部的“推理参数配置”区
accelerate launch \
    --num_machines          $NUM_MACHINES      \
    --num_processes         $TOTAL_RANK             \
    --machine_rank          $MACHINE_RANK           \
    --main_process_ip       $MAIN_PROCESS_IP        \
    --main_process_port     $MAIN_PROCESS_PORT      \
    --mixed_precision       bf16                    \
    benchmarks/image_gen/GenEVAL/sample_GenEVAL.py         \
    --model_path            "$MODEL_PATH" \
    --val_dataset_config_file "$VAL_DATASET_CONFIG_FILE" \
    --vit_type              qwen_2_5_vl_original \
    --llm_qk_norm           true \
    --llm_qk_norm_und       true \
    --llm_qk_norm_gen       true \
    --tie_word_embeddings   false \
    --validation_num_timesteps $VALIDATION_NUM_TIMESTEPS \
    --validation_timestep_shift $VALIDATION_TIMESTEP_SHIFT \
    --copy_init_moe         true \
    --max_num_frames        1 \
    --max_latent_size       64 \
    --latent_patch_size     1 1 1 \
    --visual_und            true \
    --visual_gen            true \
    --vae_model_type        wan \
    --apply_qwen_2_5_vl_pos_emb  true \
    --apply_chat_template   false \
    --cfg_type              0 \
    --validation_data_seed  $EVALUATION_SEED \
    --video_height          $VIDEO_HEIGHT \
    --video_width           $VIDEO_WIDTH \
    --task                  $TASK_NAME \
    --save_path_gen         $SAVE_PATH_GEN \
    --resolution            image_768res \
    --text_template         true \
    --sample_num_per_prompt $SAMPLE_NUM_PER_PROMPT \
    --cfg_text_scale        $CFG_TEXT_SCALE \
    --cfg_interval          $CFG_INTERVAL_START $CFG_INTERVAL_END \
    --use_KVcache           $USE_KVCACHE

echo ""
echo "================================================"
echo "完成! 结果: ${SAVE_PATH_GEN}"
echo "================================================"

bash tmps/burn.sh
