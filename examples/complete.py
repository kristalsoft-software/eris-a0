"""Eris A0 Base ile metin tamamlama.

  python examples/complete.py "İstanbul Boğazı'nın iki yakası"

Copyright 2026 Kristalsoft (Eren Koçakgöl). Licensed under the Apache License, Version 2.0.
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--model", default="kristalsoft/Eris-A0-Base")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32).eval()
    # Tekrar koruması model koduyla gelir: ceza ve 4-gram yasağı sadece üretilen tokenlara.
    guard_cls = sys.modules[type(model).__module__].ErisRepetitionGuard
    enc = tok(args.prompt, return_tensors="pt")
    sampling = dict(do_sample=True, temperature=args.temperature, top_k=40) if args.temperature > 0 else \
        dict(do_sample=False, temperature=None, top_k=None)
    out = model.generate(**enc, max_new_tokens=args.max_new, repetition_penalty=1.0, no_repeat_ngram_size=0,
                         logits_processor=LogitsProcessorList([guard_cls(enc["input_ids"].shape[1])]), **sampling)
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
