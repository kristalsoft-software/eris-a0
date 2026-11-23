"""Resmi lm-evaluation-harness görevleri: TurkishMMLU + Belebele-TR, 5-shot, seed 42.

Her model (Eris dahil) lm-eval'in "hf" modeliyle, aynı görev tanımları ve ayarlarla ölçülür.

  python -m evaluation.lmeval_run --model eris-a0-base
  python -m evaluation.lmeval_run --model kumru-2b --batch-size 8
  python -m evaluation.lmeval_run --model eris-a0-base --limit 5        # hızlı deneme
  python -m evaluation.lmeval_run --model ./my-finetune                 # yerel klasör (Eris tabanlı)

Sonuç: results/lmeval/<ad>_<n>shot.json ve ekrana tablo.

Copyright 2026 Kristalsoft (Eren Koçakgöl). Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

DEFAULT_TASKS = "turkishmmlu,belebele_tur_Latn"
CHANCE = {"turkishmmlu": 0.20, "belebele_tur_Latn": 0.25}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="evaluation/models.py'deki anahtar, HF kimliği ya da klasör")
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--num-fewshot", type=int, default=5)
    ap.add_argument("--limit", type=float, default=None, help="görev başına soru sayısı (deneme için)")
    ap.add_argument("--batch-size", default="8", help="lm-eval batch_size ('auto' olabilir)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/lmeval")
    args = ap.parse_args()

    import lm_eval
    import torch
    from lm_eval.utils import make_table

    from .models import MODELS

    spec = MODELS.get(args.model)
    hf_id = spec.hf_id if spec else args.model
    remote = spec.remote_code if spec else True
    name = args.model.strip("./").replace("/", "__")
    cuda = torch.cuda.is_available()
    dtype = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if cuda else "float32"
    model_args = f"pretrained={hf_id},dtype={dtype}" + (",trust_remote_code=True" if remote else "")
    device = args.device or ("cuda" if cuda else "cpu")
    print(f"model: {hf_id} | dtype {dtype} | {device}", flush=True)

    t0 = time.time()
    res = lm_eval.simple_evaluate(
        model="hf", model_args=model_args, batch_size=args.batch_size, device=device,
        tasks=args.tasks.split(","), num_fewshot=args.num_fewshot, limit=args.limit,
        random_seed=args.seed, numpy_random_seed=args.seed, torch_random_seed=args.seed,
        fewshot_random_seed=args.seed, log_samples=False,
    )
    dt = time.time() - t0
    print(make_table(res))

    # TurkishMMLU bir "tag" (grup değil): ortalamayı kendimiz alıyoruz (konu başına eşit ağırlık).
    summary = {}
    mmlu = [v["acc,none"] for k, v in res["results"].items() if k.startswith("turkishmmlu_")]
    if mmlu:
        summary["turkishmmlu"] = sum(mmlu) / len(mmlu)
    for k, v in res["results"].items():
        if not k.startswith("turkishmmlu_"):
            summary[k] = v["acc,none"]
    print("\nÖZET (acc):")
    for k, v in summary.items():
        print(f"  {k:<22} %{v * 100:5.1f}   (şans %{CHANCE.get(k, 0) * 100:.0f})")
    print(f"Süre: {dt / 60:.1f} dk")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{args.num_fewshot}shot{'_limit' if args.limit else ''}.json"
    keep = {k: res[k] for k in ("results", "groups", "n-shot", "n-samples", "versions") if k in res}
    keep["summary"] = summary
    keep["eris"] = {"hf_id": hf_id, "name": name, "tasks": args.tasks, "num_fewshot": args.num_fewshot,
                    "limit": args.limit, "seed": args.seed, "seconds": round(dt)}
    path.write_text(json.dumps(keep, ensure_ascii=False, indent=2, default=str))
    print("kaydedildi:", path)


if __name__ == "__main__":
    main()
