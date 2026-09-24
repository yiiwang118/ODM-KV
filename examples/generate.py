"""Generate a response with ODM-KV; no benchmark datasets are required."""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvquant import ODMPress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--backend", choices=("reference", "native"), default="reference")
    parser.add_argument("--target-avg-bits", type=float, default=2.0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.backend == "native" and not torch.cuda.is_available():
        parser.error("the native backend requires an NVIDIA CUDA GPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    user_prompt = args.prompt or (
        "The project codename is ORCHID.\n\n"
        + "The archive contains meeting notes, equipment inventories, and routine progress updates. " * 100
        + "\nWhat is the project codename? Reply with the codename only."
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype,
        attn_implementation="flash_attention_2" if args.backend == "native" else "sdpa",
    ).to(device).eval()
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
    else:
        prompt = user_prompt
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if args.backend == "native":
        from kvquant.runtime.generate import graph_generate
        eos = model.generation_config.eos_token_id
        # Preserve every model-defined stop token (e.g. Llama end-of-turn).
        tokens = graph_generate(
            model, ids, args.max_new_tokens,
            press_cfg={"target_avg_bits": args.target_avg_bits}, eos_token_id=eos,
        )
    else:
        with torch.inference_mode(), ODMPress(target_avg_bits=args.target_avg_bits)(model):
            tokens = model.generate(
                ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )[:, ids.shape[1]:]
    print(tokenizer.decode(tokens[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
