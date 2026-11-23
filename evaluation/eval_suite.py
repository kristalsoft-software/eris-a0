"""Karşılaştırma testleri: Eris A0'ı aynı boyut sınıfındaki Türkçe ve çok dilli modellerle aynı yöntemle ölçer.

Sonuçlar results/evals/<model>.json dosyalarına yazılır (görev görev, kaldığı yerden devam eder).
Tablo: results/evals/TABLO.md

  python -m evaluation.eval_suite --models eris-a0-base qwen3-0.6b tr-gpt2-medium
  python -m evaluation.eval_suite --models eris-a0-base --path ./my-finetune   # yerel klasör
  python -m evaluation.eval_suite --limit 50          # hızlı deneme
  python -m evaluation.eval_suite --table             # sadece tabloyu yeniden yaz
  python -m evaluation.eval_suite --list              # modeller ve görevler

Copyright 2026 Kristalsoft (Eren Koçakgöl). Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from . import models as M
from .scoring import fertility, run_task
from .tasks import DEFAULT_TASKS, TASKS

OUT = Path("results/evals")
# Varsayılan karşılaştırma: erişim izni gerektirmeyen, A0 boyutuna yakın modeller.
DEFAULT_MODELS = ["eris-a0-base", "tr-gpt2-medium", "tr-gpt2-large", "kanarya-750m", "hamza-large",
                  "smollm2-360m", "qwen3-0.6b"]


def run_model(key: str, tasks: list[str], limit: int | None, device: str, path: str | None, force: bool):
    t0 = time.time()
    lm = M.load(key, device, path)
    name = key
    path = OUT / f"{name}.json"
    res = json.loads(path.read_text()) if path.exists() and not force else {}
    res.update({"model": name, "key": key, "params_m": round(lm.n_params / 1e6, 1), "info": lm.info,
                "device": device})
    res.setdefault("tasks", {})
    print(f"\n=== {name} | {lm.n_params / 1e6:.0f}M parametre | {device} | yükleme {time.time() - t0:.0f} sn",
          flush=True)

    if "fertility" not in res:
        from .tasks import flores_en_bpb, flores_tr_bpb

        res["fertility"] = {"tr": fertility(lm, [d["text"] for d in flores_tr_bpb()]),
                            "en": fertility(lm, [d["text"] for d in flores_en_bpb()])}
        print(f"  kelime başına token: tr {res['fertility']['tr']:.2f} | en {res['fertility']['en']:.2f}", flush=True)

    for tn in tasks:
        task = TASKS[tn]
        want = limit if limit is not None else task.limit
        prev = res["tasks"].get(tn)
        if prev and prev.get("limit") == want:
            continue
        try:
            r = run_task(lm, task, want)
        except Exception as e:  # noqa: BLE001
            print(f"  {tn}: HATA {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        r["limit"] = want
        res["tasks"][tn] = r
        extra = f"acc {r['acc']:.3f} acc_norm {r['acc_norm']:.3f}" if task.kind == "mc" else (
            f"bpb {r['bpb']:.3f} (token başına {r['bytes_per_token']:.2f} bayt)" if task.kind == "bpb"
            else f"acc {r['acc']:.3f} ppl {r['ppl']:.1f}")
        print(f"  {tn:<15} n={r['n']:<5} {extra} | {r['seconds']:.0f} sn", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    del lm
    gc.collect()


def write_table() -> str:
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    if not rows:
        return "sonuç yok"
    # A0 önce, sonra parametre sayısına göre
    rows.sort(key=lambda r: (not r["model"].startswith("eris"), r.get("params_m") or 0))
    names = [r["model"] for r in rows]
    lines = ["| Görev | rastgele | " + " | ".join(names) + " |",
             "|---|---|" + "---|" * len(names),
             "| parametre (M) | | " + " | ".join(f"{r.get('params_m', 0):.0f}" for r in rows) + " |"]
    for lang in ("tr", "en"):
        lines.append(f"| kelime başına token ({lang}) ↓ | | " + " | ".join(
            f"{r['fertility'][lang]:.2f}" if "fertility" in r else "" for r in rows) + " |")
    for tn, task in TASKS.items():
        vals = [r["tasks"].get(tn, {}).get("main") for r in rows]
        if all(v is None for v in vals):
            continue
        lower = task.kind == "bpb"
        present = [v for v in vals if v is not None]
        best = min(present) if lower else max(present)
        cells = []
        for v in vals:
            if v is None:
                cells.append("")
            else:
                s = f"{v:.3f}" if lower else f"{100 * v:.1f}"
                cells.append(f"**{s}**" if v == best and len(present) > 1 else s)
        base = "" if task.random_baseline is None else f"{100 * task.random_baseline:.0f}"
        label = f"{tn} {'↓' if lower else '↑'}"
        lines.append(f"| {label} | {base} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    notes = ["", "↑ doğruluk (%, yüksek iyi). ↓ bit/bayt veya kelime başına token (düşük iyi).",
             "Çoktan seçmeli görevler cloze biçiminde, uzunluğa göre normalize (winogrande hariç)."]
    text = md + "\n" + "\n".join(notes) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "TABLO.md").write_text(text)
    return text


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--limit", type=int, default=None, help="Her görevde en fazla bu kadar örnek (deneme için)")
    ap.add_argument("--path", default=None, help="Tek model ölçülürken Hub yerine yerel klasör")
    ap.add_argument("--device", default=M.default_device())
    ap.add_argument("--force", action="store_true", help="Önceki sonuçları yok say")
    ap.add_argument("--table", action="store_true", help="Sadece tabloyu yeniden yaz")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT, help="Sonuç klasörü")
    args = ap.parse_args()
    OUT = args.out

    if args.list:
        for k, m in M.MODELS.items():
            print(f"model  {k:<16} {m.hf_id:<36} {m.note}")
        for k, t in TASKS.items():
            print(f"görev  {k:<16} {t.kind:<7} {t.lang:<5} {t.note}")
        return
    if not args.table:
        for key in args.models:
            try:
                run_model(key, args.tasks, args.limit, args.device, args.path, args.force)
            except Exception as e:  # noqa: BLE001
                print(f"\n=== {key}: YÜKLENEMEDİ {type(e).__name__}: {str(e)[:300]}", flush=True)
    print("\n" + write_table())


if __name__ == "__main__":
    main()
