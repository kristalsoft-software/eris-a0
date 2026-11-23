"""Terminalde Eris A0 Instruct ile sohbet.

  python examples/chat.py                       # sohbet (çıkış: boş satır)
  python examples/chat.py -q "Fotosentez nedir?" # tek soru
  python examples/chat.py --model ./my-finetune  # yerel klasör

Copyright 2026 Kristalsoft (Eren Koçakgöl). Licensed under the Apache License, Version 2.0.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kristalsoft/Eris-A0-Instruct")
    ap.add_argument("-q", "--question", action="append", help="tek seferlik soru (birden fazla verilebilir)")
    ap.add_argument("--history", action="store_true", help="önceki mesajları da gönder (model çok turlu sohbette zayıf)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32).eval()
    kw = dict(temperature=args.temperature, max_new_tokens=args.max_new)

    if args.question:
        for q in args.question:
            print(f"S: {q}\nC: {model.chat(tok, q, **kw)}\n")
        return
    messages = []
    while True:
        q = input("Sen: ").strip()
        if not q:
            break
        msgs = messages + [{"role": "user", "content": q}] if args.history else [{"role": "user", "content": q}]
        answer = model.chat(tok, msgs, **kw)
        print(f"Eris: {answer}\n")
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
