from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
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
from safetensors.torch import load_file
from transformers import set_seed
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

from common.utils.logging import get_logger
from common.utils.misc import AutoEncoderParams, tuple_mul
from config.config_factory import DataArguments, InferenceArguments, ModelArguments
from data.data_utils import add_special_tokens
from data.dataset_base import DataConfig, simple_custom_collate
from data.datasets_custom import ValidationDataset
from inference_lance import (
    PROMPT_JSON_FILENAME,
    apply_inference_defaults,
    clean_memory,
    init_from_model_path_if_needed,
    save_prompt_results,
    validate_on_fixed_batch,
)
from modeling.lance import Lance, LanceConfig, Qwen2ForCausalLM
from modeling.qwen2 import Qwen2Tokenizer
from modeling.qwen2.modeling_qwen2 import Qwen2Config
from modeling.vae.wan.model import WanVideoVAE
from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel


REPO_ROOT = Path(__file__).resolve().parent
GRADIO_TMP_ROOT = REPO_ROOT / "tmps" / "gradio_t2v_v2t"
TMP_INPUT_DIR = GRADIO_TMP_ROOT / "inputs"
RESULTS_ROOT = GRADIO_TMP_ROOT / "results"
GLOBAL_RECORDS_FILE = GRADIO_TMP_ROOT / "generation_records.jsonl"
RUN_RECORD_FILENAME = "generation_record.json"

DEFAULT_MODEL_PATH = REPO_ROOT / "downloads" / "Lance_3B_Video"
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
DEFAULT_GPUS = "0"
DEFAULT_QUEUE_SIZE = 32
USE_KVCACHE = True
TEXT_TEMPLATE = True
RECORD_WRITE_LOCK = threading.Lock()

TASK_T2V = "t2v"
TASK_V2T = "v2t"
TASK_X2T = "x2t"
TASK_X2T_VIDEO = "x2t_video"
TASK_CHOICES = [TASK_T2V, TASK_V2T]
VIDEO_RESOLUTION_CHOICES = ["video_192p", "video_360p", "video_480p"]
V2T_SYSTEM_PROMPT = "Watch the video carefully and answer the question."


def ensure_dirs() -> None:
    TMP_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def save_generation_record(record: dict, save_dir: Path) -> None:
    ensure_dirs()
    run_record_path = save_dir / RUN_RECORD_FILENAME
    with run_record_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    with RECORD_WRITE_LOCK:
        with GLOBAL_RECORDS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_seed(seed: int) -> int:
    return random.randint(0, 2**31 - 1) if seed == -1 else seed


def normalize_task(task: str) -> str:
    task = (task or DEFAULT_TASK).strip().lower()
    if task == TASK_V2T:
        return TASK_X2T_VIDEO
    if task == TASK_X2T:
        return TASK_X2T_VIDEO
    if task not in {TASK_T2V, TASK_X2T_VIDEO}:
        raise ValueError(f"Unsupported task type: {task}")
    return task


def create_request_json(task: str, prompt: str, input_video: Optional[str], question: str) -> Path:
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prompt_file = TMP_INPUT_DIR / f"{task}_{timestamp}.json"

    if task == TASK_T2V:
        payload = {"000000.mp4": prompt}
    elif task == TASK_X2T_VIDEO:
        if not input_video:
            raise ValueError("The v2t task requires an input video.")
        payload = {
            "000000": {
                "interleave_array": [input_video, [V2T_SYSTEM_PROMPT, question, ""]],
                "element_dtype_array": ["video", "text"],
                "istarget_in_interleave": [0, 1],
            }
        }
    else:
        raise ValueError(f"Unsupported task type: {task}")

    with prompt_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return prompt_file


def build_save_dir(task: str) -> Path:
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_ROOT / f"{task}_{timestamp}_{int(time.time() * 1000) % 1000:03d}"


def find_generated_video(save_dir: Path) -> Optional[Path]:
    videos = sorted(save_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return videos[0] if videos else None


def extract_text_result(save_dir: Path) -> str:
    prompt_result_path = save_dir / PROMPT_JSON_FILENAME
    if not prompt_result_path.exists():
        return ""
    with prompt_result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return ""
    first_value = next(iter(data.values()))
    return first_value if isinstance(first_value, str) else json.dumps(first_value, ensure_ascii=False)


class LanceT2VV2TPipeline:
    def __init__(self, device_id: int) -> None:
        self._init_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self.initialized = False
        self.device = device_id
        self.logger = get_logger(f"lance_t2v_v2t_gpu{device_id}")

        self.model: Optional[Lance] = None
        self.vae_model: Optional[WanVideoVAE] = None
        self.vae_config: Optional[AutoEncoderParams] = None
        self.tokenizer: Optional[Qwen2Tokenizer] = None
        self.new_token_ids: Optional[dict] = None
        self.image_token_id: Optional[int] = None
        self.base_model_args: Optional[ModelArguments] = None
        self.base_data_args: Optional[DataArguments] = None
        self.base_inference_args: Optional[InferenceArguments] = None

    def _log_stage(self, stage_name: str, start_time: float, extra: str = "") -> None:
        elapsed = time.perf_counter() - start_time
        suffix = f" | {extra}" if extra else ""
        print(f"[startup][gpu:{self.device}] {stage_name} done in {elapsed:.2f}s{suffix}", flush=True)

    def _build_base_model_args(self) -> ModelArguments:
        model_path = str(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else ""
        return ModelArguments(
            model_path=model_path,
            vit_type=DEFAULT_VIT_TYPE,
            llm_qk_norm=True,
            llm_qk_norm_und=True,
            llm_qk_norm_gen=True,
            tie_word_embeddings=False,
            max_num_frames=121,
            max_latent_size=64,
            latent_patch_size=[1, 1, 1],
        )

    def _build_base_inference_args(self) -> InferenceArguments:
        return InferenceArguments(
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
            save_path_gen=str(RESULTS_ROOT),
            resolution=DEFAULT_RESOLUTION,
            text_template=TEXT_TEMPLATE,
            use_KVcache=USE_KVCACHE,
        )

    def initialize(self) -> None:
        with self._init_lock:
            if self.initialized:
                return

            ensure_dirs()
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Lance T2V/V2T Gradio requires a GPU environment.")
            if self.device >= torch.cuda.device_count():
                raise RuntimeError(
                    f"GPU {self.device} is unavailable. Detected {torch.cuda.device_count()} GPU(s)."
                )
            torch.cuda.set_device(self.device)

            model_args = self._build_base_model_args()
            data_args = DataArguments()
            inference_args = self._build_base_inference_args()
            apply_inference_defaults(model_args, data_args, inference_args)
            inference_args.validation_noise_seed = inference_args.validation_data_seed

            self.base_model_args = model_args
            self.base_data_args = data_args
            self.base_inference_args = inference_args

            set_seed(inference_args.global_seed)

            stage_start = time.perf_counter()
            print(
                f"[startup][gpu:{self.device}] Loading LLM config: {Path(model_args.model_path) / 'llm_config.json'}",
                flush=True,
            )
            llm_config: Qwen2Config = Qwen2Config.from_json_file(str(Path(model_args.model_path) / "llm_config.json"))
            self._log_stage("LLM config load", stage_start)

            llm_config.layer_module = model_args.layer_module
            llm_config.qk_norm = model_args.llm_qk_norm
            llm_config.qk_norm_und = model_args.llm_qk_norm_und
            llm_config.qk_norm_gen = model_args.llm_qk_norm_gen
            llm_config.tie_word_embeddings = model_args.tie_word_embeddings
            llm_config.freeze_und = inference_args.freeze_und
            llm_config.apply_qwen_2_5_vl_pos_emb = inference_args.apply_qwen_2_5_vl_pos_emb

            stage_start = time.perf_counter()
            print(f"[startup][gpu:{self.device}] Initializing LLM weights: {model_args.model_path}", flush=True)
            language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)
            self._log_stage("LLM weight init", stage_start)

            vit_model = None
            vit_config = None
            if inference_args.visual_und:
                if model_args.vit_type not in ("qwen2_5_vl", "qwen_2_5_vl_original"):
                    raise ValueError(f"Unsupported vit_type: {model_args.vit_type}")
                stage_start = time.perf_counter()
                print(f"[startup][gpu:{self.device}] Loading VIT config: {model_args.vit_path}", flush=True)
                vit_config = Qwen2_5_VLVisionConfig.from_pretrained(model_args.vit_path)
                self._log_stage("VIT config load", stage_start)

                stage_start = time.perf_counter()
                print(
                    f"[startup][gpu:{self.device}] Loading VIT weights: {Path(model_args.vit_path) / 'vit.safetensors'}",
                    flush=True,
                )
                vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
                vit_weights = load_file(str(Path(model_args.vit_path) / "vit.safetensors"))
                vit_model.load_state_dict(vit_weights, strict=True)
                self._log_stage("VIT weight load", stage_start)
                clean_memory(vit_weights)

            if inference_args.visual_gen:
                stage_start = time.perf_counter()
                print(f"[startup][gpu:{self.device}] Initializing VAE", flush=True)
                vae_model = WanVideoVAE()
                vae_config = deepcopy(vae_model.vae_config)
                self._log_stage("VAE init", stage_start)
            else:
                vae_model = None
                vae_config = None

            config = LanceConfig(
                visual_gen=inference_args.visual_gen,
                visual_und=inference_args.visual_und,
                llm_config=llm_config,
                vit_config=vit_config if inference_args.visual_und else None,
                vae_config=vae_config if inference_args.visual_gen else None,
                latent_patch_size=model_args.latent_patch_size,
                max_num_frames=model_args.max_num_frames,
                max_latent_size=model_args.max_latent_size,
                vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
                connector_act=model_args.connector_act,
                interpolate_pos=model_args.interpolate_pos,
                timestep_shift=inference_args.timestep_shift,
            )
            model: Lance = Lance(
                language_model=language_model,
                vit_model=vit_model if inference_args.visual_und else None,
                vit_type=model_args.vit_type,
                config=config,
                training_args=inference_args,
            )

            stage_start = time.perf_counter()
            print(f"[startup][gpu:{self.device}] Moving Lance model to GPU {self.device}", flush=True)
            model = model.to(self.device)
            self._log_stage("Lance model move to GPU", stage_start)

            stage_start = time.perf_counter()
            print(f"[startup][gpu:{self.device}] Loading tokenizer: {model_args.model_path}", flush=True)
            tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
            tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
            self._log_stage("tokenizer load and special token init", stage_start, extra=f"num_new_tokens={num_new_tokens}")

            if inference_args.copy_init_moe:
                language_model.init_moe()

            init_from_model_path_if_needed(model, model_args)

            if num_new_tokens > 0:
                model.language_model.resize_token_embeddings(len(tokenizer))
                model.config.llm_config.vocab_size = len(tokenizer)
                model.language_model.config.vocab_size = len(tokenizer)

            if model_args.vit_type.lower() == "qwen2_5_vl":
                from common.model.hacks import hack_qwen2_5_vl_config

                language_model = hack_qwen2_5_vl_config(language_model)

            image_token_id = language_model.config.video_token_id
            new_token_ids.update({"image_token_id": image_token_id})
            model.update_tokenizer(tokenizer=tokenizer)

            if model_args.tie_word_embeddings:
                model.language_model.untie_lm_head()
                model.language_model.copy_new_token_rows_to_lm_head(num_new_tokens)
                model_args.tie_word_embeddings = False
                llm_config.tie_word_embeddings = False
            else:
                assert (
                    model.language_model.get_input_embeddings().weight.data.data_ptr()
                    != model.language_model.get_output_embeddings().weight.data.data_ptr()
                ), "tie_word_embeddings conflict"

            model = model.to(device=self.device, dtype=torch.bfloat16)
            model.eval()
            if vae_model is not None and hasattr(vae_model, "eval"):
                vae_model.eval()

            self.model = model
            self.vae_model = vae_model
            self.vae_config = vae_config
            self.tokenizer = tokenizer
            self.new_token_ids = new_token_ids
            self.image_token_id = image_token_id
            self.initialized = True
            print(f"[startup][gpu:{self.device}] Lance T2V/V2T Gradio model loaded and ready for reuse.", flush=True)

    def _build_request_batch(
        self,
        prompt_file: Path,
        model_args: ModelArguments,
        data_args: DataArguments,
        inference_args: InferenceArguments,
    ):
        assert self.tokenizer is not None
        assert self.new_token_ids is not None
        assert self.vae_config is not None

        dataset_config = DataConfig.from_yaml(str(prompt_file))
        if inference_args.visual_und:
            dataset_config.vit_patch_size = model_args.vit_patch_size
            dataset_config.vit_patch_size_temporal = model_args.vit_patch_size_temporal
            dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side
        if inference_args.visual_gen:
            vae_downsample = tuple_mul(
                tuple(model_args.latent_patch_size),
                (
                    self.vae_config.downsample_temporal,
                    self.vae_config.downsample_spatial,
                    self.vae_config.downsample_spatial,
                ),
            )
            dataset_config.latent_patch_size = model_args.latent_patch_size
            dataset_config.vae_downsample = vae_downsample
            dataset_config.max_latent_size = model_args.max_latent_size
            dataset_config.max_num_frames = model_args.max_num_frames

        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

        dataset_config.num_frames = inference_args.num_frames
        dataset_config.H = inference_args.video_height
        dataset_config.W = inference_args.video_width
        dataset_config.task = inference_args.task
        dataset_config.resolution = inference_args.resolution
        dataset_config.text_template = inference_args.text_template

        val_dataset = ValidationDataset(
            jsonl_path=str(prompt_file),
            tokenizer=self.tokenizer,
            data_args=data_args,
            model_args=model_args,
            training_args=inference_args,
            new_token_ids=self.new_token_ids,
            dataset_config=dataset_config,
            local_rank=0,
            world_size=1,
        )
        return simple_custom_collate([val_dataset[0]])

    def generate(
        self,
        task: str,
        prompt: str,
        input_video: Optional[str],
        question: str,
        height: int,
        width: int,
        num_frames: int,
        seed: int,
        resolution: str,
        validation_num_timesteps: int,
        validation_timestep_shift: float,
        cfg_text_scale: float,
    ):
        self.initialize()
        internal_task = normalize_task(task)
        prompt = (prompt or "").strip()
        question = (question or "").strip()
        input_video = str(input_video).strip() if input_video else ""

        if internal_task == TASK_T2V and not prompt:
            return None, "", "Please enter a prompt.", ""
        if internal_task == TASK_X2T_VIDEO and not question:
            return None, "", "Please enter a question.", ""
        if internal_task == TASK_X2T_VIDEO and not input_video:
            return None, "", "Please upload an input video.", ""
        if height <= 0 or width <= 0:
            return None, "", "Height and width must be greater than 0.", ""
        if num_frames <= 0:
            return None, "", "The number of frames must be greater than 0.", ""

        assert self.model is not None
        assert self.tokenizer is not None
        assert self.new_token_ids is not None
        assert self.image_token_id is not None
        assert self.base_model_args is not None
        assert self.base_data_args is not None
        assert self.base_inference_args is not None

        with self._generate_lock:
            torch.cuda.set_device(self.device)
            actual_seed = normalize_seed(int(seed))
            prompt_file = create_request_json(
                task=internal_task,
                prompt=prompt,
                input_video=input_video,
                question=question,
            )
            save_dir = build_save_dir(internal_task)
            save_dir.mkdir(parents=True, exist_ok=True)
            request_started_at = datetime.now().isoformat(timespec="seconds")

            request_model_args = deepcopy(self.base_model_args)
            request_model_args.cfg_text_scale = float(cfg_text_scale)

            request_data_args = deepcopy(self.base_data_args)
            request_data_args.val_dataset_config_file = str(prompt_file)

            request_inference_args = deepcopy(self.base_inference_args)
            request_inference_args.validation_num_timesteps = int(validation_num_timesteps)
            request_inference_args.validation_timestep_shift = float(validation_timestep_shift)
            request_inference_args.validation_data_seed = actual_seed
            request_inference_args.validation_noise_seed = actual_seed
            request_inference_args.video_height = int(height)
            request_inference_args.video_width = int(width)
            request_inference_args.num_frames = int(num_frames)
            request_inference_args.resolution = resolution
            request_inference_args.save_path_gen = str(save_dir)
            request_inference_args.task = internal_task
            request_inference_args.text_template = TEXT_TEMPLATE
            request_inference_args.prompt_data_dict = {}

            try:
                print(
                    "[lance_gradio_t2v_v2t] Start generation "
                    f"| task={internal_task} | gpu={self.device} | seed={actual_seed} | "
                    f"size={height}x{width} | frames={num_frames} | resolution={resolution}",
                    flush=True,
                )
                val_data_cpu = self._build_request_batch(
                    prompt_file=prompt_file,
                    model_args=request_model_args,
                    data_args=request_data_args,
                    inference_args=request_inference_args,
                )
                generate_start = time.perf_counter()
                validate_on_fixed_batch(
                    fsdp_model=self.model,
                    vae_model=self.vae_model,
                    tokenizer=self.tokenizer,
                    val_data_cpu=val_data_cpu,
                    training_args=request_inference_args,
                    model_args=request_model_args,
                    inference_args=request_inference_args,
                    new_token_ids=self.new_token_ids,
                    image_token_id=self.image_token_id,
                    device=self.device,
                    save_source_video=False,
                    save_path_gen=request_inference_args.save_path_gen,
                    save_path_gt="",
                )
                elapsed = time.perf_counter() - generate_start
                save_prompt_results(request_inference_args.prompt_data_dict, request_inference_args.save_path_gen, self.logger)
                clean_memory()

                video_path = find_generated_video(save_dir) if internal_task == TASK_T2V else None
                text_result = extract_text_result(save_dir) if internal_task == TASK_X2T_VIDEO else ""
                record = {
                    "request_started_at": request_started_at,
                    "request_finished_at": datetime.now().isoformat(timespec="seconds"),
                    "status": "success",
                    "task": internal_task,
                    "gpu": self.device,
                    "prompt": prompt,
                    "question": question,
                    "input_video": input_video,
                    "seed": actual_seed,
                    "height": int(height),
                    "width": int(width),
                    "num_frames": int(num_frames),
                    "resolution": resolution,
                    "validation_num_timesteps": int(validation_num_timesteps),
                    "validation_timestep_shift": float(validation_timestep_shift),
                    "cfg_text_scale": float(cfg_text_scale),
                    "elapsed_seconds": round(elapsed, 3),
                    "prompt_file": str(prompt_file),
                    "output_dir": str(save_dir),
                    "video_path": str(video_path) if video_path is not None else "",
                    "text_result": text_result,
                }
                if internal_task == TASK_T2V and video_path is None:
                    record["status"] = "completed_without_video"
                if internal_task == TASK_X2T_VIDEO and not text_result:
                    record["status"] = "completed_without_text"
                save_generation_record(record, save_dir)

                logs = "\n".join(
                    [
                        "[lance_gradio_t2v_v2t] Generation finished in-process.",
                        f"task={internal_task}",
                        f"gpu={self.device}",
                        f"seed={actual_seed}",
                        f"height={height}",
                        f"width={width}",
                        f"num_frames={num_frames}",
                        f"resolution={resolution}",
                        f"validation_num_timesteps={validation_num_timesteps}",
                        f"validation_timestep_shift={validation_timestep_shift}",
                        f"cfg_text_scale={cfg_text_scale}",
                        f"elapsed={elapsed:.2f}s",
                        f"output_dir={save_dir}",
                    ]
                )

                if internal_task == TASK_T2V:
                    if video_path is None:
                        status = (
                            "Inference completed, but no generated video was found.\n\n"
                            f"- Task: `{internal_task}`\n"
                            f"- GPU: `{self.device}`\n"
                            f"- Actual seed: `{actual_seed}`\n"
                            f"- Output directory: `{save_dir}`"
                        )
                        return None, "", status, logs
                    status = (
                        "Inference completed.\n\n"
                        f"- Task: `{internal_task}`\n"
                        f"- GPU: `{self.device}`\n"
                        f"- Actual seed: `{actual_seed}`\n"
                        f"- Output directory: `{save_dir}`\n"
                        f"- Result file: `{video_path}`"
                    )
                    return str(video_path), "", status, logs

                status = (
                    "Understanding completed.\n\n"
                    f"- Task: `{task}`\n"
                    f"- GPU: `{self.device}`\n"
                    f"- Actual seed: `{actual_seed}`\n"
                    f"- Output directory: `{save_dir}`"
                )
                return None, text_result, status, logs
            except Exception:
                error_trace = traceback.format_exc()
                print(error_trace, flush=True)
                record = {
                    "request_started_at": request_started_at,
                    "request_finished_at": datetime.now().isoformat(timespec="seconds"),
                    "status": "failed",
                    "task": internal_task,
                    "gpu": self.device,
                    "prompt": prompt,
                    "question": question,
                    "input_video": input_video,
                    "seed": actual_seed,
                    "height": int(height),
                    "width": int(width),
                    "num_frames": int(num_frames),
                    "resolution": resolution,
                    "validation_num_timesteps": int(validation_num_timesteps),
                    "validation_timestep_shift": float(validation_timestep_shift),
                    "cfg_text_scale": float(cfg_text_scale),
                    "prompt_file": str(prompt_file),
                    "output_dir": str(save_dir),
                    "video_path": "",
                    "text_result": "",
                    "error": error_trace,
                }
                save_generation_record(record, save_dir)
                status = (
                    "Inference failed.\n\n"
                    f"- Task: `{internal_task}`\n"
                    f"- GPU: `{self.device}`\n"
                    f"- Actual seed: `{actual_seed}`\n"
                    f"- Output directory: `{save_dir}`"
                )
                return None, "", status, error_trace


class PipelinePool:
    def __init__(self, gpu_ids: list[int]) -> None:
        if not gpu_ids:
            raise ValueError("At least one GPU must be configured.")
        self.gpu_ids = gpu_ids
        self.pipelines = [LanceT2VV2TPipeline(device_id=gpu_id) for gpu_id in gpu_ids]
        self._available = deque(self.pipelines)
        self._condition = threading.Condition()

    @property
    def size(self) -> int:
        return len(self.pipelines)

    @property
    def gpu_summary(self) -> str:
        return ",".join(str(gpu_id) for gpu_id in self.gpu_ids)

    def initialize_all(self) -> None:
        print(f"[startup] Preparing parallel GPU preload: {self.gpu_ids}", flush=True)
        exceptions: list[Exception] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.size) as executor:
            futures = {
                executor.submit(pipeline.initialize): pipeline.device for pipeline in self.pipelines
            }
            for future in concurrent.futures.as_completed(futures):
                gpu_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[startup][gpu:{gpu_id}] Preload failed: {exc}", flush=True)
                    exceptions.append(exc)
        if exceptions:
            raise RuntimeError(f"Preload failed on {len(exceptions)} GPU(s). Please check the terminal logs.") from exceptions[0]
        print(f"[startup] GPU preload finished. Ready to handle {self.size} concurrent request(s).", flush=True)

    def acquire(self) -> LanceT2VV2TPipeline:
        with self._condition:
            while not self._available:
                self._condition.wait()
            return self._available.popleft()

    def release(self, pipeline: LanceT2VV2TPipeline) -> None:
        with self._condition:
            self._available.append(pipeline)
            self._condition.notify()

    def generate(
        self,
        task: str,
        prompt: str,
        input_video: Optional[str],
        question: str,
        height: int,
        width: int,
        num_frames: int,
        seed: int,
        resolution: str,
        validation_num_timesteps: int,
        validation_timestep_shift: float,
        cfg_text_scale: float,
    ):
        pipeline = self.acquire()
        try:
            return pipeline.generate(
                task=task,
                prompt=prompt,
                input_video=input_video,
                question=question,
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed,
                resolution=resolution,
                validation_num_timesteps=validation_num_timesteps,
                validation_timestep_shift=validation_timestep_shift,
                cfg_text_scale=cfg_text_scale,
            )
        finally:
            self.release(pipeline)


PIPELINE_POOL: Optional[PipelinePool] = None
QUEUE_MAX_SIZE = DEFAULT_QUEUE_SIZE


def run_task(
    task: str,
    prompt: str,
    input_video: Optional[str],
    question: str,
    height: int,
    width: int,
    num_frames: int,
    seed: int,
    resolution: str,
    validation_num_timesteps: int,
    validation_timestep_shift: float,
    cfg_text_scale: float,
):
    assert PIPELINE_POOL is not None
    return PIPELINE_POOL.generate(
        task=task,
        prompt=prompt,
        input_video=input_video,
        question=question,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=seed,
        resolution=resolution,
        validation_num_timesteps=validation_num_timesteps,
        validation_timestep_shift=validation_timestep_shift,
        cfg_text_scale=cfg_text_scale,
    )


def build_status_markdown() -> str:
    gpu_text = "unknown"
    concurrency = 1
    if PIPELINE_POOL is not None:
        gpu_text = PIPELINE_POOL.gpu_summary
        concurrency = PIPELINE_POOL.size
    return (
        f"**Status**  GPU: `{gpu_text}`  |  Max concurrency: `{concurrency}`  |  "
        f"Queue limit: `{QUEUE_MAX_SIZE}`  |  Preload mode: `parallel`"
    )


def update_task_ui(task: str):
    task = (task or DEFAULT_TASK).strip().lower()
    if task == TASK_T2V:
        return (
            gr.update(label="Prompt", placeholder="Describe the video you want to generate...", visible=True),
            gr.update(label="Input Video", visible=False, value=None),
            gr.update(label="Question", placeholder="Please enter a question", visible=False, value=""),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(value=""),
        )
    return (
        gr.update(label="Prompt", placeholder="This task does not require a prompt", visible=False, value=""),
        gr.update(label="Input Video", visible=True),
        gr.update(label="Question", placeholder="Describe the question you want the model to answer", visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=""),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Lance T2V/V2T Gradio") as demo:
        gr.Markdown(
            """
            # Lance T2V/V2T

            Supports two tasks: `t2v` and `v2t`.
            `v2t` is mapped to the internal `x2t_video` task in the backend.
            The service preloads one model per GPU at startup, and requests are automatically dispatched to idle GPUs.
            """
        )
        gr.Markdown(build_status_markdown())

        with gr.Row():
            with gr.Column(scale=1):
                task = gr.Dropdown(label="Task", choices=TASK_CHOICES, value=DEFAULT_TASK)
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=6,
                    placeholder="Describe the video you want to generate...",
                )
                input_video = gr.Video(label="Input Video", visible=False)
                question = gr.Textbox(
                    label="Question",
                    lines=3,
                    placeholder="Describe the question you want the model to answer",
                    visible=False,
                )
                with gr.Row():
                    height = gr.Slider(
                        minimum=192,
                        maximum=1024,
                        step=16,
                        value=DEFAULT_HEIGHT,
                        label="Height",
                    )
                    width = gr.Slider(
                        minimum=192,
                        maximum=1024,
                        step=16,
                        value=DEFAULT_WIDTH,
                        label="Width",
                    )
                num_frames = gr.Slider(
                    minimum=1,
                    maximum=121,
                    step=1,
                    value=DEFAULT_NUM_FRAMES,
                    label="Output Frames",
                )
                seed = gr.Number(
                    label="Seed",
                    value=DEFAULT_BASIC_SEED,
                    precision=0,
                    info="-1 means using a random seed each time",
                )
                resolution = gr.Dropdown(
                    label="RESOLUTION",
                    choices=VIDEO_RESOLUTION_CHOICES,
                    value=DEFAULT_RESOLUTION,
                )

                with gr.Accordion("Advanced Parameters", open=False):
                    validation_num_timesteps = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=DEFAULT_TIMESTEPS,
                        label="VALIDATION_NUM_TIMESTEPS",
                    )
                    validation_timestep_shift = gr.Number(
                        label="VALIDATION_TIMESTEP_SHIFT",
                        value=DEFAULT_TIMESTEP_SHIFT,
                    )
                    cfg_text_scale = gr.Number(
                        label="CFG_TEXT_SCALE",
                        value=DEFAULT_CFG_TEXT_SCALE,
                    )

                run_button = gr.Button("Run", variant="primary")

            with gr.Column(scale=1):
                output_video = gr.Video(label="Video Result")
                output_text = gr.Textbox(label="Text Result", lines=8)
                status = gr.Markdown("Waiting to run.")
                logs = gr.Textbox(label="Run Logs", lines=22, max_lines=30)

        task.change(
            fn=update_task_ui,
            inputs=[task],
            outputs=[prompt, input_video, question, height, width, num_frames, output_text],
        )

        run_button.click(
            fn=run_task,
            inputs=[
                task,
                prompt,
                input_video,
                question,
                height,
                width,
                num_frames,
                seed,
                resolution,
                validation_num_timesteps,
                validation_timestep_shift,
                cfg_text_scale,
            ],
            outputs=[output_video, output_text, status, logs],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lance T2V/V2T Gradio")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--gpus",
        default=DEFAULT_GPUS,
        help="Comma-separated GPU list, for example: 0,1,2,3,4,5,6",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=DEFAULT_QUEUE_SIZE,
        help="Maximum number of queued Gradio requests.",
    )
    return parser.parse_args()


def parse_gpu_ids(gpu_string: str) -> list[int]:
    gpu_ids: list[int] = []
    for item in gpu_string.split(","):
        item = item.strip()
        if not item:
            continue
        gpu_ids.append(int(item))
    if not gpu_ids:
        raise ValueError("No valid GPU IDs were parsed.")
    return gpu_ids


if __name__ == "__main__":
    args = parse_args()
    QUEUE_MAX_SIZE = args.queue_size
    gpu_ids = parse_gpu_ids(args.gpus)
    PIPELINE_POOL = PipelinePool(gpu_ids)
    PIPELINE_POOL.initialize_all()
    demo = build_demo()
    demo.queue(
        max_size=args.queue_size,
        default_concurrency_limit=PIPELINE_POOL.size,
    ).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
