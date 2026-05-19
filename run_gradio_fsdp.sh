#!/usr/bin/env bash
set -e
cd /home/as-ad/image-models/bytedance-research/Lance-Github
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/as-ad/qwen32b-training-data/.venv/bin/torchrun --nproc_per_node=2 lance_gradio_fsdp.py --share >> /tmp/lance_gradio.log 2>&1
