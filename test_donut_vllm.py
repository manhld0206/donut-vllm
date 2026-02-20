#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams

from donut_vllm_v1.plugin import register as _register_donut


def _load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _clean_prediction(text: str, task_prompt: str, tokenizer) -> str:
    cleaned = text
    if task_prompt and cleaned.startswith(task_prompt):
        cleaned = cleaned[len(task_prompt) :]
    if tokenizer.eos_token:
        cleaned = cleaned.replace(tokenizer.eos_token, "")
    if tokenizer.pad_token:
        cleaned = cleaned.replace(tokenizer.pad_token, "")
    cleaned = cleaned.strip()
    if cleaned and cleaned[0] == "<":
        cleaned = re.sub(r"^<[^>]+>", "", cleaned).strip()
    return cleaned


def main() -> None:
    from transformers import DonutProcessor

    parser = argparse.ArgumentParser(
        description="Smoke test for Donut on vLLM (VisionEncoderDecoder)."
    )
    parser.add_argument(
        "--model",
        default="naver-clova-ix/donut-base-finetuned-cord-v2",
        help="HF model id or local path.",
    )
    parser.add_argument(
        "--image",
        default="./example.png",
        help="Path to a local image file.",
    )
    parser.add_argument(
        "--task-prompt",
        default="<s_cord-v2>",
        help="Task prompt token for the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="vLLM dtype setting.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.6,
        help="vLLM GPU memory utilization target.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Override max model length in vLLM.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup generations before timing.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timed generations for the benchmark.",
    )
    args = parser.parse_args()

    _register_donut()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    prep_start = time.perf_counter()
    processor = DonutProcessor.from_pretrained(
        args.model,
        use_fast=True,
    )
    image = _load_image(image_path)

    prompt_token_ids = (
        processor.tokenizer(
            args.task_prompt, add_special_tokens=False, return_tensors="pt"
        )
        .input_ids[0]
        .tolist()
    )

    max_tokens = 128
    prep_time = time.perf_counter() - prep_start

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        # stop_token_ids=None,
        ignore_eos=True,
        # bad_words=None,
    )

    llm_init_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        limit_mm_per_prompt={"image": 1},
        # trust_remote_code=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # max_model_len=768,
        # max_num_seqs=1,
        # hf_overrides={"architectures": ["DonutForConditionalGeneration"]},
        hf_overrides={"decoder_start_token_id": prompt_token_ids[0]},
    )
    llm_init_time = time.perf_counter() - llm_init_start

    prompts = [{
        "prompt": "",
        "multi_modal_data": {"image": image},
    }]

    warmup_runs = max(args.warmup, 0)
    for _ in range(warmup_runs):
        llm.generate(prompts, sampling_params, use_tqdm=False)

    timings: list[float] = []
    for _ in range(max(args.runs, 1)):
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        timings.append(time.perf_counter() - start_time)
        output = outputs[0].outputs[0]

    print(f"Prep time (s) processor+image+prompt={prep_time:.4f}")
    print(f"LLM init time (s)={llm_init_time:.4f}")
    if timings:
        avg = sum(timings) / len(timings)
        p50 = sorted(timings)[len(timings) // 2]
        print(f"Generate time (s) avg={avg:.4f} p50={p50:.4f} runs={len(timings)}")

    # print(len(output))
    token_ids = output.token_ids
    print(token_ids)
    print(len(token_ids))
    cleaned = _clean_prediction(output.text, args.task_prompt, processor.tokenizer)
    structured = processor.token2json(cleaned)

    print("=== Raw ===")
    print(output.text)
    print("=== Cleaned ===")
    print(cleaned)
    print("=== JSON ===")
    print(structured)


if __name__ == "__main__":
    main()
