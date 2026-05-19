#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/benchmarks/sample_env.sh"

# ========================= Inference Parameters =========================
NUM_GPUS=2

TASK_NAME=x2t_image # t2i | image_edit | t2v | video_edit | x2t_image | x2t_video
USE_IMAGE_MODEL=true  # set true for image tasks, false for video tasks

VALIDATION_NUM_TIMESTEPS=30 # 50
VALIDATION_TIMESTEP_SHIFT=3.5
VALIDATION_DATA_SEED=42
CFG_TEXT_SCALE=4.0
USE_KVCACHE=true

NUM_FRAMES=50             # max: 121 frames, unused for image tasks
VIDEO_HEIGHT=768          # unused for editing
VIDEO_WIDTH=768           # unused for editing
TEXT_TEMPLATE=true

# Auto-select model path based on task type
case "$TASK_NAME" in
    t2i|image_edit|x2t_image)
        MODEL_PATH="downloads/lance_3b"
        RESOLUTION="image_768res"
        VISUAL_GEN=true
        VISUAL_UND=true
        ;;
    t2v|video_edit|x2t_video)
        MODEL_PATH="downloads/lance_3b_video"
        RESOLUTION="video_480p"
        VISUAL_GEN=true
        VISUAL_UND=true
        ;;
esac

# ========================= Auto-generated Paths =========================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
KVCACHE_TAG=""
if [ "$USE_KVCACHE" = "true" ]; then
    KVCACHE_TAG="_kvcache"
fi
SAVE_PATH_GEN="results/${TASK_NAME}_sample_ts${VALIDATION_NUM_TIMESTEPS}_tts${VALIDATION_TIMESTEP_SHIFT}_seed${VALIDATION_DATA_SEED}_cfg${CFG_TEXT_SCALE}${KVCACHE_TAG}_${TIMESTAMP}"

if [ -z "$MODEL_PATH" ]; then
    echo "Error: please set MODEL_PATH manually in the configuration section at the top of this script."
    exit 1
fi

# ============================== Environment and Distributed Setup ==============================
lance_setup_common_env
lance_setup_distributed_env "$NUM_GPUS"
lance_setup_shard_env 1

# ========================= Show Task Configuration =========================
echo "================================================"
echo "Lance Inference"
echo "================================================"
echo "Task: ${TASK_NAME}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Save path: ${SAVE_PATH_GEN}"
echo "Resolution: ${VIDEO_HEIGHT}x${VIDEO_WIDTH}"
echo "Output frames: ${NUM_FRAMES}"
echo "Model path: ${MODEL_PATH}"
echo ""
echo "Key parameters:"
echo "  - validation_num_timesteps: ${VALIDATION_NUM_TIMESTEPS}"
echo "  - validation_timestep_shift: ${VALIDATION_TIMESTEP_SHIFT}"
echo "  - validation_data_seed: ${VALIDATION_DATA_SEED}"
echo "  - cfg_text_scale: ${CFG_TEXT_SCALE}"
echo "  - num_frames: ${NUM_FRAMES}"
echo "  - use_KVcache: ${USE_KVCACHE}"
echo "================================================"
echo ""

# ============================== Run Inference ==============================
PYTHON_VENV="$HOME/qwen32b-training-data/.venv"
TORCHRUN="$PYTHON_VENV/bin/torchrun"
if [ ! -f "$TORCHRUN" ]; then
    TORCHRUN="torchrun"
fi

$TORCHRUN \
    --nproc_per_node=$NUM_GPUS \
    inference_lance.py \
    --model_path            "$MODEL_PATH" \
    --vit_type              qwen_2_5_vl_original \
    --llm_qk_norm           true \
    --llm_qk_norm_und       true \
    --llm_qk_norm_gen       true \
    --tie_word_embeddings   false \
    --validation_num_timesteps $VALIDATION_NUM_TIMESTEPS \
    --validation_timestep_shift $VALIDATION_TIMESTEP_SHIFT \
    --copy_init_moe         true \
    --max_num_frames        121 \
    --max_latent_size       64 \
    --latent_patch_size     1 1 1 \
    --visual_und            $VISUAL_UND \
    --visual_gen            $VISUAL_GEN \
    --vae_model_type        wan \
    --apply_qwen_2_5_vl_pos_emb true \
    --apply_chat_template   false \
    --cfg_type              0 \
    --validation_data_seed  $VALIDATION_DATA_SEED \
    --video_height          $VIDEO_HEIGHT \
    --video_width           $VIDEO_WIDTH \
    --num_frames            $NUM_FRAMES \
    --task                  $TASK_NAME \
    --save_path_gen         "$SAVE_PATH_GEN" \
    --resolution            "$RESOLUTION" \
    --text_template         "$TEXT_TEMPLATE" \
    --cfg_text_scale        $CFG_TEXT_SCALE \
    --use_KVcache           "$USE_KVCACHE"

echo ""
echo "================================================"
echo "Done! Results: ${SAVE_PATH_GEN}"
echo "================================================"
