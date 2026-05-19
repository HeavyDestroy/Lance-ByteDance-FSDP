#!/usr/bin/env python3
"""
Lance Gradio demo with FSDP dual-GPU — launch via torchrun.

Usage:
  cd /path/to/Lance-Github
  ~/qwen32b-training-data/.venv/bin/torchrun --nproc_per_node=2 \
    lance_gradio_fsdp.py --share

Architecture:
  Rank 0 runs the Gradio HTTP server.
  All ranks participate in FSDP sharded inference via NCCL broadcast sync.
  The main loop polls for work; rank 0's Gradio handler enqueues requests.

Memory strategy:
  VIT and VAE are kept on CPU by default. They are moved to GPU only when
  needed (VAE for encode/decode steps, VIT for understanding tasks).
  This keeps GPU memory free for the LLM forward pass (the hungriest part).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
import traceback
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import set_seed
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

from common.utils.logging import get_logger
from common.utils.misc import AutoEncoderParams, tuple_mul
from config.config_factory import DataArguments, InferenceArguments, ModelArguments, get_model_path
from data.data_utils import add_special_tokens
from data.dataset_base import DataConfig, simple_custom_collate
from data.datasets_custom import ValidationDataset
from inference_lance import (
    PROMPT_JSON_FILENAME,
    GENERATION_TASKS,
    UNDERSTANDING_TASKS,
    TASK_T2I,
    TASK_IMAGE_EDIT,
    TASK_X2T_IMAGE,
    apply_inference_defaults,
    clean_memory,
    init_from_model_path_if_needed,
    save_prompt_results,
    save_understanding_results,
    validate_on_fixed_batch,
)
from modeling.lance import Lance, LanceConfig, Qwen2ForCausalLM
from modeling.qwen2 import Qwen2Tokenizer
from modeling.qwen2.modeling_qwen2 import Qwen2Config
from modeling.vae.wan.model import WanVideoVAE
from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel

# ─── Distributed config ───────────────────────────────────────────────
assert "RANK" in os.environ and "WORLD_SIZE" in os.environ, "Must run with torchrun"
dist.init_process_group("nccl")
GLOBAL_RANK = dist.get_rank()
WORLD_SIZE = dist.get_world_size()
LOCAL_RANK = GLOBAL_RANK % torch.cuda.device_count()
DEVICE = LOCAL_RANK
torch.cuda.set_device(DEVICE)

IS_RANK0 = GLOBAL_RANK == 0
log_rank0 = print if IS_RANK0 else (lambda *_, **__: None)

# ─── Constants ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
GRADIO_TMP_ROOT = REPO_ROOT / "tmps" / "gradio_fsdp"
TMP_INPUT_DIR = GRADIO_TMP_ROOT / "inputs"
RESULTS_ROOT = GRADIO_TMP_ROOT / "results"
GLOBAL_RECORDS_FILE = GRADIO_TMP_ROOT / "generation_records.jsonl"
RUN_RECORD_FILENAME = "generation_record.json"

DEFAULT_MODEL_PATH = REPO_ROOT / "downloads" / "lance_3b_video"
DEFAULT_VIT_TYPE = "qwen_2_5_vl_original"
DEFAULT_TASK = "t2v"
DEFAULT_TIMESTEPS = 30
DEFAULT_TIMESTEP_SHIFT = 3.5
DEFAULT_CFG_TEXT_SCALE = 4.0
DEFAULT_RESOLUTION = "video_480p"
DEFAULT_BASIC_SEED = -1
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 848
DEFAULT_NUM_FRAMES = 50
DEFAULT_QUEUE_SIZE = 16
USE_KVCACHE = True
TEXT_TEMPLATE = True
RECORD_WRITE_LOCK = threading.Lock()

TASK_T2V = "t2v"
TASK_V2T = "v2t"
TASK_X2T = "x2t"
TASK_X2T_VIDEO = "x2t_video"

# System prompts
V2T_SYSTEM_PROMPT = "Please describe the content of the given video."
I2T_SYSTEM_PROMPT = "Please describe the content of the given image."

TASK_CHOICES = [
    TASK_T2V,
    TASK_T2I,
    TASK_IMAGE_EDIT,
    TASK_V2T,
    TASK_X2T_IMAGE,
]
VIDEO_RESOLUTION_CHOICES = [
    "video_480p",
    "video_720p",
    "video_1080p",
]


# ─── Shared state (rank 0 only) ───────────────────────────────────────
_request_queue: queue.Queue = queue.Queue()
_result_container: dict = {}
_result_event = threading.Event()


# ─── VAE / VIT device management helpers ──────────────────────────────
VAE_ON_GPU = False
VIT_ON_GPU = False
_vae_lock = threading.Lock()

def move_vae_to_gpu(vae_model=None):
    """Move VAE to GPU on this rank. Call before vae_encode/vae_decode."""
    global VAE_ON_GPU
    if vae_model is None or VAE_ON_GPU:
        return
    if hasattr(vae_model, "to"):
        log_rank0(f"[mem] Moving VAE to cuda:{DEVICE}")
        vae_model.to(device=f"cuda:{DEVICE}")
    VAE_ON_GPU = True
    torch.cuda.empty_cache()

def move_vae_to_cpu(vae_model=None):
    """Move VAE to CPU to free GPU memory. Call after vae_encode/vae_decode."""
    global VAE_ON_GPU
    if vae_model is None or not VAE_ON_GPU:
        return
    if hasattr(vae_model, "to"):
        log_rank0(f"[mem] Moving VAE to CPU")
        vae_model.to(device="cpu")
    VAE_ON_GPU = False
    torch.cuda.empty_cache()

def move_vit_to_gpu(vit_model=None):
    """Move VIT to GPU. Call before x2t inference."""
    global VIT_ON_GPU
    if vit_model is None or VIT_ON_GPU:
        return
    if hasattr(vit_model, "to"):
        log_rank0(f"[mem] Moving VIT to cuda:{DEVICE} (bf16)")
        vit_model.to(device=f"cuda:{DEVICE}", dtype=torch.bfloat16)
    VIT_ON_GPU = True
    torch.cuda.empty_cache()

def move_vit_to_cpu(vit_model=None):
    """Move VIT to CPU. Call after x2t inference."""
    global VIT_ON_GPU
    if vit_model is None or not VIT_ON_GPU:
        return
    if hasattr(vit_model, "to"):
        log_rank0(f"[mem] Moving VIT to CPU")
        vit_model.to(device="cpu")
    VIT_ON_GPU = False
    torch.cuda.empty_cache()


# ─── Model init (all ranks) ───────────────────────────────────────────
def init_fsdp_model(
    model_path: Path = DEFAULT_MODEL_PATH,
) -> tuple[Lance, Optional[WanVideoVAE], AutoEncoderParams, Qwen2Tokenizer, dict, int]:
    """Load model, VIT, VAE, tokenizer on all ranks. VIT and VAE stay on CPU."""

    log_rank0(f"[init] FSDP init on {WORLD_SIZE} GPUs (rank {GLOBAL_RANK}, local {LOCAL_RANK})")

    # ── LLM config ──
    stage_start = time.perf_counter()
    log_rank0(f"[init] Loading LLM config: {model_path / 'llm_config.json'}")
    llm_config: Qwen2Config = Qwen2Config.from_json_file(str(Path(model_path) / "llm_config.json"))
    log_rank0(f"[init] LLM config load done in {time.perf_counter() - stage_start:.2f}s")

    default_model_args = ModelArguments(model_path=str(model_path), vit_type=DEFAULT_VIT_TYPE)
    llm_config.layer_module = default_model_args.layer_module
    llm_config.qk_norm = True
    llm_config.qk_norm_und = True
    llm_config.qk_norm_gen = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    llm_config.apply_qwen_2_5_vl_pos_emb = True

    # ── LLM weights (~3B) ──
    stage_start = time.perf_counter()
    log_rank0(f"[init] Initializing LLM weights (~3B)")
    language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)
    log_rank0(f"[init] LLM weight init done in {time.perf_counter() - stage_start:.2f}s")

    # ── VIT ──
    stage_start = time.perf_counter()
    vit_path = get_model_path("vit.qwen2_5_vl")
    log_rank0(f"[init] Loading VIT from {vit_path}")
    vit_config = Qwen2_5_VLVisionConfig.from_pretrained(vit_path)
    vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
    vit_weights_path = str(Path(vit_path) / "vit.safetensors")
    if os.path.exists(vit_weights_path):
        vit_weights = load_file(vit_weights_path)
        vit_model.load_state_dict(vit_weights, strict=True)
        # Free VIT weight dict after loading
        clean_memory(vit_weights)
    else:
        log_rank0(f"[init] WARNING: {vit_weights_path} not found, using uninitialized VIT")
    log_rank0(f"[init] VIT load done in {time.perf_counter() - stage_start:.2f}s")

    # ── VAE (always init, keep on CPU) ──
    stage_start = time.perf_counter()
    log_rank0(f"[init] Initializing VAE")
    vae_model = WanVideoVAE()
    vae_config: AutoEncoderParams = deepcopy(vae_model.vae_config)
    log_rank0(f"[init] VAE init done in {time.perf_counter() - stage_start:.2f}s")

    # ── Lance config ──
    config = LanceConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=[1, 1, 1],
        max_num_frames=121,
        max_latent_size=64,
        vit_max_num_patch_per_side=-1,
        connector_act="silu",
        interpolate_pos=True,
        timestep_shift=3.5,
    )

    # ── Create Lance model ──
    model: Lance = Lance(
        language_model=language_model,
        vit_model=vit_model,
        vit_type=DEFAULT_VIT_TYPE,
        config=config,
        training_args=InferenceArguments(visual_gen=True, visual_und=True, vae_model_type="wan"),
    )

    stage_start = time.perf_counter()
    log_rank0(f"[init] Converting model to bf16 (CPU)")
    model.to(dtype=torch.bfloat16)
    log_rank0(f"[init] bf16 conversion done in {time.perf_counter() - stage_start:.2f}s")

    # ── Tokenizer ──
    stage_start = time.perf_counter()
    log_rank0(f"[init] Loading tokenizer")
    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    log_rank0(f"[init] Tokenizer done in {time.perf_counter() - stage_start:.2f}s")

    # ── MoE init ──
    if True:  # copy_init_moe
        language_model.init_moe()

    # ── Load checkpoint ──
    init_from_model_path_if_needed(model, ModelArguments(model_path=str(model_path), vit_type=DEFAULT_VIT_TYPE))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    image_token_id = language_model.config.video_token_id
    new_token_ids.update({"image_token_id": image_token_id})
    model.update_tokenizer(tokenizer=tokenizer)

    assert model.language_model.get_input_embeddings().weight.data.data_ptr() != \
           model.language_model.get_output_embeddings().weight.data.data_ptr(), \
           "tie_word_embeddings conflict"

    model.eval()

    # ═══════ FSDP Wrapping ═══════
    stage_start = time.perf_counter()
    log_rank0(f"[init] FSDP-wrapping {len(model.language_model.model.layers)} decoder layers")

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    lm = model.language_model

    # Move non-layer components to GPU (stay full on each rank)
    lm.lm_head = lm.lm_head.to(device=f"cuda:{DEVICE}", dtype=torch.bfloat16)
    lm.model.embed_tokens = lm.model.embed_tokens.to(device=f"cuda:{DEVICE}", dtype=torch.bfloat16)
    lm.model.norm = lm.model.norm.to(device=f"cuda:{DEVICE}", dtype=torch.bfloat16)
    if hasattr(lm.model, 'norm_moe_gen') and lm.model.norm_moe_gen is not None:
        lm.model.norm_moe_gen = lm.model.norm_moe_gen.to(device=f"cuda:{DEVICE}", dtype=torch.bfloat16)
    if hasattr(lm.model, 'rotary_emb') and lm.model.rotary_emb is not None:
        lm.model.rotary_emb = lm.model.rotary_emb.to(device=f"cuda:{DEVICE}")

    # Wrap each decoder layer with FSDP FULL_SHARD
    for i in range(len(lm.model.layers)):
        lm.model.layers[i] = FSDP(
            lm.model.layers[i],
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
        )

    log_rank0(f"[init] {len(lm.model.layers)} decoder layers FSDP-wrapped in {time.perf_counter() - stage_start:.2f}s")

    # Move remaining Lance components to GPU (connector, embedders)
    if hasattr(model, 'vit_model') and model.vit_model is not None:
        # VIT stays on CPU — moved on demand
        log_rank0(f"[init] VIT kept on CPU (moved to GPU on demand)")
        model.vit_model = model.vit_model.to(device="cpu", dtype=torch.bfloat16)
    if hasattr(model, 'time_embedder'):
        model.time_embedder = model.time_embedder.to(device=f"cuda:{DEVICE}")
    if hasattr(model, 'connector') and model.connector is not None:
        model.connector = model.connector.to(device=f"cuda:{DEVICE}", dtype=torch.bfloat16)
    for attr in ['vae2llm', 'llm2vae', 'latent_pos_embed']:
        if hasattr(model, attr):
            getattr(model, attr).to(device=f"cuda:{DEVICE}")

    # VAE stays on CPU — moved on demand for encode/decode
    if hasattr(vae_model, "eval"):
        vae_model.eval()
    # Move VAE's scale tensor to CPU too. Wan2_2_VAE is a plain object (not nn.Module),
    # so calling .to("cpu") on the inner model doesn't propagate to its .scale attribute.
    # Without this, CPU decode fails with "Expected all tensors to be on the same device".
    if hasattr(vae_model, 'vae') and hasattr(vae_model.vae, 'scale'):
        _s = vae_model.vae.scale
        if isinstance(_s, torch.Tensor):
            vae_model.vae.scale = _s.cpu()
        elif isinstance(_s, (list, tuple)):
            vae_model.vae.scale = type(_s)(
                x.cpu() if isinstance(x, torch.Tensor) else x for x in _s
            )
    log_rank0(f"[init] VAE kept on CPU (moved to GPU on demand)")

    # Clear stale CUDA allocator cache after init
    torch.cuda.empty_cache()
    log_rank0(f"[init] Cache cleared. Model ready.")
    dist.barrier()

    return model, vae_model, vae_config, tokenizer, new_token_ids, image_token_id


# ─── Wrapper: validate_on_fixed_batch with VAE/VIT management ─────────
def run_inference_managed(
    fsdp_model: Lance,
    vae_model: Optional[WanVideoVAE],
    vit_model,
    tokenizer, batch, training_args, model_args, inference_args,
    new_token_ids, image_token_id, device, save_path_gen,
    save_source_video=False, save_path_gt="",
) -> None:
    """Wrapper around validate_on_fixed_batch with VAE/VIT GPU management."""

    task = inference_args.task

    if task in GENERATION_TASKS:
        # --- Generation tasks (t2v, t2i, etc.) ---
        # VIT: not needed for generation, keep on CPU
        move_vit_to_cpu(vit_model)
        # VAE: needed only for decode at end; kept on CPU during LLM forward
        move_vae_to_cpu(vae_model)

        validate_on_fixed_batch(
            fsdp_model=fsdp_model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            val_data_cpu=batch,
            training_args=training_args,
            model_args=model_args,
            inference_args=inference_args,
            new_token_ids=new_token_ids,
            image_token_id=image_token_id,
            device=device,
            save_source_video=save_source_video,
            save_path_gen=save_path_gen,
            save_path_gt=save_path_gt,
        )

    elif task in UNDERSTANDING_TASKS:
        # --- Understanding tasks (x2t, v2t) ---
        move_vit_to_gpu(vit_model)
        move_vae_to_gpu(vae_model)

        validate_on_fixed_batch(
            fsdp_model=fsdp_model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            val_data_cpu=batch,
            training_args=training_args,
            model_args=model_args,
            inference_args=inference_args,
            new_token_ids=new_token_ids,
            image_token_id=image_token_id,
            device=device,
            save_source_video=save_source_video,
            save_path_gen=save_path_gen,
            save_path_gt=save_path_gt,
        )

        move_vit_to_cpu(vit_model)
        move_vae_to_cpu(vae_model)

    clean_memory()


# ─── Utility functions ────────────────────────────────────────────────
def ensure_dirs():
    for d in [GRADIO_TMP_ROOT, TMP_INPUT_DIR, RESULTS_ROOT]:
        d.mkdir(parents=True, exist_ok=True)


def normalize_task(task: str) -> str:
    task = (task or "").strip().lower()
    if task in ("t2v", "text2video", "text-to-video"):
        return TASK_T2V
    if task in ("t2i", "text2image", "text-to-image"):
        return TASK_T2I
    if task in ("image_edit", "i2i", "img2img", "image-to-image"):
        return TASK_IMAGE_EDIT
    if task in ("v2t", "video2text", "video-to-text"):
        return TASK_V2T
    if task in ("x2t_image", "i2t", "image2text", "image-to-text"):
        return TASK_X2T_IMAGE
    if task in ("x2t_video", "vqa"):
        return TASK_X2T_VIDEO
    return TASK_T2V


def normalize_seed(seed: int) -> int:
    if seed < 0:
        return random.randint(0, 2**31 - 1)
    return seed


def build_save_dir(task: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return RESULTS_ROOT / f"{task}_{timestamp}"


def create_request_json(task, prompt, input_video, input_image, question) -> Path:
    """Write a one-sample JSON file for the dataset loader.
    Format matches Lance's expected input: {filename: prompt, ...} for t2v/t2i,
    or interleave dict for understanding/editing tasks."""
    import yaml
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prompt_file = TMP_INPUT_DIR / f"{task}_{timestamp}.json"
    if task == TASK_T2V:
        payload = {"000000.mp4": prompt.strip()}
    elif task == TASK_T2I:
        payload = {"000000.png": prompt.strip()}
    elif task == TASK_IMAGE_EDIT:
        # image_edit: input image + text instruction → edited image
        if not input_image:
            raise ValueError("Input image required for image_edit")
        payload = {
            "000000": {
                "interleave_array": [prompt.strip(), input_image, input_image],
                "element_dtype_array": ["text", "image", "image"],
                "istarget_in_interleave": [0, 0, 1],
            }
        }
    elif task == TASK_X2T_IMAGE:
        if not input_image:
            raise ValueError("Input image required for x2t_image")
        payload = {
            "000000": {
                "interleave_array": [input_image, [I2T_SYSTEM_PROMPT, question or "", ""]],
                "element_dtype_array": ["image", "text"],
                "istarget_in_interleave": [0, 1],
            }
        }
    elif task == TASK_V2T or task == TASK_X2T_VIDEO:
        payload = {
            "000000": {
                "interleave_array": [input_video or "", [V2T_SYSTEM_PROMPT, question or "", ""]],
                "element_dtype_array": ["video", "text"],
                "istarget_in_interleave": [0, 1],
            }
        }
    else:
        raise ValueError(f"Unsupported task: {task}")
    with prompt_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return prompt_file


def save_generation_record(record: dict, save_dir: Path):
    with RECORD_WRITE_LOCK:
        with open(save_dir / RUN_RECORD_FILENAME, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        with open(GLOBAL_RECORDS_FILE, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_generated_video(save_dir: Path) -> Optional[str]:
    for f in sorted(save_dir.iterdir()):
        if f.suffix in (".mp4", ".gif", ".png", ".jpg"):
            return str(f.resolve())
    return None


def extract_text_result(save_dir: Path) -> str:
    json_path = save_dir / "result.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", str(data[0]))
            if isinstance(data, dict):
                return data.get("generated_text", str(data))
    return ""


# ─── Gradio UI (rank 0 only) ──────────────────────────────────────────
def run_task_handler(
    task, prompt, input_video, input_image, question, height, width, num_frames,
    seed, resolution, validation_num_timesteps, validation_timestep_shift, cfg_text_scale,
):
    """Called by Gradio on rank 0 in a background thread."""
    global _result_container, _result_event

    request_data = dict(
        task=task, prompt=prompt, input_video=input_video, input_image=input_image, question=question,
        height=height, width=width, num_frames=num_frames, seed=seed,
        resolution=resolution, validation_num_timesteps=validation_num_timesteps,
        validation_timestep_shift=validation_timestep_shift,
        cfg_text_scale=cfg_text_scale,
        prompt_data_dict={},
    )

    _request_queue.put(request_data)
    _result_event.wait()
    _result_event.clear()
    return _result_container.pop("result", (None, None, "", "No result", ""))


def build_status_markdown() -> str:
    return (
        f"**Status**  GPUs: `{WORLD_SIZE}`  |  "
        f"FSDP: `FULL_SHARD`  |  Rank 0 serves UI  |  "
        f"VAE/VIT: `on-demand`"
    )


def update_task_ui(task: str):
    """Return visibility updates for all UI components based on task."""
    task = (task or DEFAULT_TASK).strip().lower()
    # Order: prompt, input_video, input_image, question, height, width, num_frames,
    #        output_text, output_video, output_image
    if task == TASK_T2V:
        return (
            gr.update(label="Prompt", placeholder="Describe the video...", visible=True),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=""),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(value=""),
            gr.update(visible=True),   # output_video
            gr.update(visible=False),  # output_image
        )
    if task == TASK_T2I:
        return (
            gr.update(label="Prompt", placeholder="Describe the image...", visible=True),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=""),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False, value=1),
            gr.update(value=""),
            gr.update(visible=False),  # output_video
            gr.update(visible=True),   # output_image
        )
    if task == TASK_IMAGE_EDIT:
        return (
            gr.update(label="Edit Instruction", placeholder="Describe the edit...", visible=True),
            gr.update(visible=False, value=None),
            gr.update(label="Input Image", visible=True),
            gr.update(visible=False, value=""),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False, value=1),
            gr.update(value=""),
            gr.update(visible=False),  # output_video
            gr.update(visible=True),   # output_image
        )
    if task == TASK_V2T or task == TASK_X2T_VIDEO:
        return (
            gr.update(visible=False, value=""),
            gr.update(label="Input Video", visible=True),
            gr.update(visible=False, value=None),
            gr.update(label="Question", placeholder="Ask about the video...", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),  # output_video
            gr.update(visible=False),  # output_image
        )
    if task == TASK_X2T_IMAGE:
        return (
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=None),
            gr.update(label="Input Image", visible=True),
            gr.update(label="Question", placeholder="Ask about the image...", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),  # output_video
            gr.update(visible=False),  # output_image
        )
    # Default: same as t2v
    return (
        gr.update(visible=True),
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=""),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(value=""),
        gr.update(visible=True),   # output_video
        gr.update(visible=False),  # output_image
    )


def clear_cache_handler():
    """Clear CUDA cache and temp dirs, then return status."""
    import shutil
    torch.cuda.empty_cache()
    for d in [TMP_INPUT_DIR, RESULTS_ROOT]:
        if d.exists():
            shutil.rmtree(str(d))
        d.mkdir(parents=True, exist_ok=True)
    log_rank0("[cache] Cleared CUDA cache and temp dirs")
    return "Cache cleared."


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Lance FSDP Multi-Task") as demo:
        gr.Markdown(
            "# Lance — FSDP Multi-Task Demo\n\n"
            f"Powered by FSDP + NCCL across {WORLD_SIZE} GPUs. "
            "Supports `t2v`, `t2i`, `image_edit`, `v2t`, `x2t_image`."
        )
        gr.Markdown(build_status_markdown())

        with gr.Row():
            with gr.Column(scale=1):
                task = gr.Dropdown(label="Task", choices=TASK_CHOICES, value=DEFAULT_TASK)
                prompt = gr.Textbox(label="Prompt", lines=6, placeholder="Describe the video to generate...")
                input_video = gr.Video(label="Input Video", visible=False)
                input_image = gr.Image(label="Input Image", type="filepath", visible=False)
                question = gr.Textbox(label="Question", lines=3, placeholder="Ask about the video...", visible=False)
                with gr.Row():
                    height = gr.Slider(minimum=192, maximum=1024, step=16, value=DEFAULT_HEIGHT, label="Height")
                    width = gr.Slider(minimum=192, maximum=1024, step=16, value=DEFAULT_WIDTH, label="Width")
                num_frames = gr.Slider(minimum=1, maximum=121, step=1, value=DEFAULT_NUM_FRAMES, label="Output Frames")
                seed = gr.Number(label="Seed", value=DEFAULT_BASIC_SEED, precision=0, info="-1 = random")
                resolution = gr.Dropdown(label="Resolution", choices=VIDEO_RESOLUTION_CHOICES, value=DEFAULT_RESOLUTION)

                with gr.Accordion("Advanced", open=False):
                    validation_num_timesteps = gr.Slider(minimum=1, maximum=100, step=1, value=DEFAULT_TIMESTEPS, label="Timesteps")
                    validation_timestep_shift = gr.Number(label="Timestep Shift", value=DEFAULT_TIMESTEP_SHIFT)
                    cfg_text_scale = gr.Number(label="CFG Scale", value=DEFAULT_CFG_TEXT_SCALE)

                with gr.Row():
                    run_button = gr.Button("Run", variant="primary")
                    clear_cache_btn = gr.Button("Clear Cache", variant="secondary")

            with gr.Column(scale=1):
                output_video = gr.Video(label="Video Result", visible=True)
                output_image = gr.Image(label="Image Result", type="filepath", visible=False)
                output_text = gr.Textbox(label="Text Result", lines=8)
                status = gr.Markdown("Ready.")
                logs = gr.Textbox(label="Logs", lines=22, max_lines=30)

        task.change(
            fn=update_task_ui,
            inputs=[task],
            outputs=[prompt, input_video, input_image, question, height, width, num_frames,
                     output_text, output_video, output_image],
        )

        run_button.click(
            fn=run_task_handler,
            inputs=[
                task, prompt, input_video, input_image, question, height, width, num_frames,
                seed, resolution, validation_num_timesteps,
                validation_timestep_shift, cfg_text_scale,
            ],
            outputs=[output_video, output_image, output_text, status, logs],
        )

        clear_cache_btn.click(
            fn=clear_cache_handler,
            inputs=[],
            outputs=[status],
        )

    return demo


# ─── Main FSDP loop (all ranks) ───────────────────────────────────────
def main():
    # Init model on all ranks
    model, vae_model, vae_config, tokenizer, new_token_ids, image_token_id = init_fsdp_model()

    # Build base InferenceArguments
    inference_args = InferenceArguments(
        validation_num_timesteps=DEFAULT_TIMESTEPS,
        validation_timestep_shift=DEFAULT_TIMESTEP_SHIFT,
        copy_init_moe=True,
        visual_und=True,
        visual_gen=True,
        vae_model_type="wan",
        apply_qwen_2_5_vl_pos_emb=True,
        apply_chat_template=False,
        cfg_type=0,
        validation_data_seed=42,
        video_height=DEFAULT_HEIGHT,
        video_width=DEFAULT_WIDTH,
        num_frames=DEFAULT_NUM_FRAMES,
        task=DEFAULT_TASK,
        resolution=DEFAULT_RESOLUTION,
        text_template=TEXT_TEMPLATE,
        use_KVcache=USE_KVCACHE,
    )
    model_args = ModelArguments(model_path=str(DEFAULT_MODEL_PATH), vit_type=DEFAULT_VIT_TYPE)
    data_args = DataArguments()

    if IS_RANK0:
        ensure_dirs()
        demo = build_demo()
        # Launch Gradio in a separate thread so main loop keeps running
        gradio_thread = threading.Thread(
            target=lambda: demo.queue(max_size=DEFAULT_QUEUE_SIZE, default_concurrency_limit=1).launch(
                server_name="0.0.0.0", server_port=7860, share=True,
            ),
            daemon=True,
        )
        gradio_thread.start()
        log_rank0("[gradio] Gradio server thread started on :7860")

    # Sync loop — all ranks participate
    log_rank0("[loop] Entering FSDP sync loop")
    dist.barrier()

    while True:
        # Rank 0: check for queued request
        request_data = None
        if IS_RANK0:
            try:
                request_data = _request_queue.get_nowait()
            except queue.Empty:
                request_data = None

        # Broadcast to all ranks: [has_request, serialized_data or None]
        payload = [request_data is not False]
        if IS_RANK0:
            if request_data is not None:
                payload = [True, json.dumps(request_data, default=str)]
            else:
                payload = [False, ""]
        else:
            payload = [False, ""]

        dist.broadcast_object_list(payload, src=0)
        has_work, serialized = payload

        if has_work and serialized:
            # ── All ranks decode request ──
            req = json.loads(serialized)

            # Build prompt json
            internal_task = normalize_task(req["task"])
            prompt_file = create_request_json(
                internal_task, req.get("prompt", ""),
                req.get("input_video", ""), req.get("input_image", ""), req.get("question", ""),
            )
            save_dir = build_save_dir(internal_task)
            save_dir.mkdir(parents=True, exist_ok=True)

            request_inference_args = deepcopy(inference_args)
            request_inference_args.validation_num_timesteps = int(req["validation_num_timesteps"])
            request_inference_args.validation_timestep_shift = float(req["validation_timestep_shift"])
            request_inference_args.validation_data_seed = normalize_seed(int(req["seed"]))
            request_inference_args.validation_noise_seed = request_inference_args.validation_data_seed
            request_inference_args.video_height = int(req["height"])
            request_inference_args.video_width = int(req["width"])
            request_inference_args.num_frames = int(req["num_frames"])
            request_inference_args.resolution = req["resolution"]
            request_inference_args.save_path_gen = str(save_dir)
            request_inference_args.task = internal_task
            request_inference_args.text_template = TEXT_TEMPLATE
            request_inference_args.prompt_data_dict = {}

            request_model_args = deepcopy(model_args)
            request_model_args.cfg_text_scale = float(req["cfg_text_scale"])

            request_data_args = deepcopy(data_args)
            request_data_args.val_dataset_config_file = str(prompt_file)

            # Build batch
            dataset_config = DataConfig.from_yaml(str(prompt_file))
            dataset_config.vit_patch_size = model_args.vit_patch_size
            dataset_config.vit_patch_size_temporal = model_args.vit_patch_size_temporal
            dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side
            vae_downsample = tuple_mul(
                (1, 1, 1),
                (vae_config.downsample_temporal, vae_config.downsample_spatial, vae_config.downsample_spatial),
            )
            dataset_config.latent_patch_size = (1, 1, 1)
            dataset_config.vae_downsample = vae_downsample
            dataset_config.max_latent_size = 64
            dataset_config.max_num_frames = 121
            dataset_config.text_cond_dropout_prob = 0.0
            dataset_config.vae_cond_dropout_prob = 0.0
            dataset_config.vit_cond_dropout_prob = 0.0
            dataset_config.num_frames = request_inference_args.num_frames
            dataset_config.H = request_inference_args.video_height
            dataset_config.W = request_inference_args.video_width
            dataset_config.task = internal_task
            dataset_config.resolution = request_inference_args.resolution
            dataset_config.text_template = TEXT_TEMPLATE

            val_dataset = ValidationDataset(
                jsonl_path=str(prompt_file),
                tokenizer=tokenizer,
                data_args=request_data_args,
                model_args=request_model_args,
                training_args=request_inference_args,
                new_token_ids=new_token_ids,
                dataset_config=dataset_config,
                local_rank=0,
                world_size=1,
            )

            batch = simple_custom_collate([val_dataset[0]])

            request_started_at = datetime.now().isoformat(timespec="seconds")

            # ── Run inference (all ranks participate via FSDP) ──
            try:
                torch.cuda.set_device(DEVICE)
                log_rank0(f"[infer] Starting {internal_task} | GPU={DEVICE} | "
                          f"size={request_inference_args.video_height}x{request_inference_args.video_width}")

                # Use the managed wrapper that handles VAE/VIT GPU placement
                run_inference_managed(
                    fsdp_model=model,
                    vae_model=vae_model,
                    vit_model=getattr(model, 'vit_model', None),
                    tokenizer=tokenizer,
                    batch=batch,
                    training_args=request_inference_args,
                    model_args=request_model_args,
                    inference_args=request_inference_args,
                    new_token_ids=new_token_ids,
                    image_token_id=image_token_id,
                    device=DEVICE,
                    save_source_video=False,
                    save_path_gen=str(save_dir),
                    save_path_gt="",
                )

                clean_memory()
                dist.barrier()

                # Gather results
                gathered = [None for _ in range(WORLD_SIZE)]
                dist.all_gather_object(gathered, request_inference_args.prompt_data_dict)
                merged = {}
                for d in gathered:
                    merged.update(d)
                request_inference_args.prompt_data_dict = merged

                if IS_RANK0:
                    save_prompt_results(request_inference_args.prompt_data_dict, str(save_dir), get_logger())
                    if internal_task in ("x2t_image", "x2t_video", "v2t", "vqa", "caption"):
                        save_understanding_results(
                            prompt_data_dict=request_inference_args.prompt_data_dict,
                            dataset_config_file=str(prompt_file),
                            save_path_gen=str(save_dir),
                        )

                if IS_RANK0:
                    video_path = find_generated_video(save_dir) if internal_task in (TASK_T2V,) else None
                    image_path = find_generated_video(save_dir) if internal_task in (TASK_T2I, TASK_IMAGE_EDIT) else None
                    text_result = extract_text_result(save_dir) if internal_task in (TASK_X2T_IMAGE, TASK_X2T_VIDEO, TASK_V2T) else ""

                    record = {
                        "request_started_at": request_started_at,
                        "request_finished_at": datetime.now().isoformat(timespec="seconds"),
                        "status": "success",
                        "task": internal_task,
                        "gpu": f"fsdp({WORLD_SIZE}gpu)",
                        "prompt": req.get("prompt", ""),
                        "question": req.get("question", ""),
                        "input_video": req.get("input_video", ""),
                        "seed": request_inference_args.validation_data_seed,
                        "height": request_inference_args.video_height,
                        "width": request_inference_args.video_width,
                        "num_frames": request_inference_args.num_frames,
                        "resolution": request_inference_args.resolution,
                        "validation_num_timesteps": request_inference_args.validation_num_timesteps,
                        "validation_timestep_shift": request_inference_args.validation_timestep_shift,
                        "cfg_text_scale": request_model_args.cfg_text_scale,
                        "prompt_file": str(prompt_file),
                        "output_dir": str(save_dir),
                        "output_path": str(video_path or image_path or ""),
                        "text_result": text_result,
                    }
                    save_generation_record(record, save_dir)

                    if internal_task == TASK_T2V:
                        result = (str(video_path) if video_path else None, None, "", f"Done: {save_dir.name}", "")
                    elif internal_task in (TASK_T2I, TASK_IMAGE_EDIT):
                        result = (None, str(image_path) if image_path else None, "", f"Done: {save_dir.name}", "")
                    elif internal_task in UNDERSTANDING_TASKS:
                        result = (None, None, text_result, f"Done: {save_dir.name}", "")
                    else:
                        result = (None, None, text_result, f"Done: {save_dir.name}", "")

                    _result_container["result"] = result
                    _result_event.set()

            except Exception as e:
                error_trace = traceback.format_exc()
                log_rank0(f"[infer] FAILED: {e}\n{error_trace}")
                if IS_RANK0:
                    _result_container["result"] = (None, None, "", f"Failed: {e}", error_trace)
                    _result_event.set()
                dist.barrier()

        else:
            # No work — brief sleep
            time.sleep(0.2)


if __name__ == "__main__":
    main()
