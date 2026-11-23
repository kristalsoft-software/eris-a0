"""Benchmark sonuçlarını tek rapora toplar.

  python -m evaluation.report                      # results/ altındakiler
  python -m evaluation.report --root /path/to/results

Kaynaklar:
  results/lmeval/*.json  (evaluation/lmeval_run.py: TurkishMMLU, Belebele-TR, 5-shot)
  results/evals/TABLO.md (evaluation/eval_suite.py: bit/bayt, kelime başına token, cloze görevleri)
Çıktı: <root>/RAPOR.md ve ekrana aynısı.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CHANCE = {"turkishmmlu": 0.20, "belebele_tur_Latn": 0.25}
LABEL = {"turkishmmlu": "TurkishMMLU (9 konu)", "belebele_tur_Latn": "Belebele-TR"}


def lmeval_table(root: Path) -> str:
    files = sorted((root / "lmeval").glob("*.json"))
    files = [f for f in files if "_limit" not in f.name] or files  # tam koşu varsa deneme koşularını gösterme
    if not files:
        return ""
    runs = []
    for f in files:
        d = json.loads(f.read_text())
        e = d.get("eris", {})
        runs.append((e.get("name") or f.stem, e.get("num_fewshot"), d.get("summary", {}), "_limit" in f.name))
    runs.sort(key=lambda r: (not r[0].startswith("eris"), r[0]))
    tasks = [t for t in CHANCE if any(t in r[2] for r in runs)]
    head = "| Görev | şans | " + " | ".join(r[0] + (" (deneme)" if r[3] else "") for r in runs) + " |"
    lines = [head, "|---|---|" + "---|" * len(runs)]
    for t in tasks:
        vals = [r[2].get(t) for r in runs]
        best = max(v for v in vals if v is not None)
        cells = ["" if v is None else (f"**{100 * v:.1f}**" if v == best and len(runs) > 1 else f"{100 * v:.1f}")
                 for v in vals]
        lines.append(f"| {LABEL.get(t, t)} | {100 * CHANCE[t]:.0f} | " + " | ".join(cells) + " |")
    shots = {r[1] for r in runs}
    note = f"\nlm-evaluation-harness, {'/'.join(str(s) for s in shots)}-shot, seed 42. Doğruluk (%), yüksek iyi.\n"
    return "\n".join(lines) + note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results"))
    args = ap.parse_args()
    parts = ["# Eris A0 benchmark raporu", ""]
    lm = lmeval_table(args.root)
    if lm:
        parts += ["## Resmi benchmark'lar (lm-eval)", "", lm]
    tablo = args.root / "evals" / "TABLO.md"
    if tablo.exists():
        parts += ["## Karşılaştırma testleri (aynı yöntem, tüm modeller)", "", tablo.read_text()]
    if len(parts) == 2:
        parts.append("Sonuç bulunamadı.")
    text = "\n".join(parts)
    (args.root / "RAPOR.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
