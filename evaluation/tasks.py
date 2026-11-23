"""Karşılaştırma testleri: Türkçe ve İngilizce çoktan seçmeli görevler + bit/bayt (BPB) ölçümleri.

Biçim: tüm çoktan seçmeli görevler "cloze" (CF) biçiminde puanlanır: her seçeneğin metni bağlamın
devamı olarak verilir ve modelin o metne verdiği log-olasılık karşılaştırılır. Harfle cevaplama
(A/B/C/D) biçimi ~1B altındaki modellerde rastgeleye yakın kalır, küçük modelleri ayırt etmez.
Bu yüzden küçük model karşılaştırmalarında (ör. HF FineWeb/SmolLM ablasyonları) CF kullanılır.

Bağlam ve seçenek kalıpları lm-evaluation-harness'taki karşılıklarına uyar (hellaswag, winogrande,
xcopa, lambada). Puanlar yine de yayınlanmış lm-eval sonuçlarıyla birebir aynı olmayabilir: önemli
olan, tüm modellerin burada AYNI yöntemle ölçülmesi.

Görev türleri:
  "mc"     : {"ctxs": [bağlam] veya seçenek başına bağlam, "conts": [devam metinleri], "gold": int}
  "greedy" : {"ctx": bağlam, "cont": devam} -> devam tokenlarının hepsi en olası token mı (lambada)
  "bpb"    : {"text": metin} -> bit/bayt (tokenizer'dan bağımsız, modeller arası karşılaştırılabilir)
"""

from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def _dl(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset")


def _parquet(repo: str, filename: str) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(_dl(repo, filename)).to_pylist()


def _lit(x):
    """Bazı veri setlerinde listeler/sözlükler metin olarak saklanmış."""
    return ast.literal_eval(x) if isinstance(x, str) else x


def _hs_pre(text: str) -> str:
    """lm-eval hellaswag ön işlemesi (Türkçe çeviri için [başlık] da)."""
    text = text.strip()
    text = text.replace(" [title]", ". ").replace(" [başlık]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


# --------------------------------------------------------------------------- görevler

def xcopa_tr():
    conn = {"cause": "çünkü", "effect": "bu yüzden"}
    out = []
    for d in _parquet("cambridgeltl/xcopa", "tr/test-00000-of-00001.parquet"):
        ctx = d["premise"].strip()[:-1] + " " + conn[d["question"]]
        conts = [" " + c[0].lower() + c[1:] for c in (d["choice1"], d["choice2"])]
        out.append({"ctxs": [ctx], "conts": conts, "gold": int(d["label"])})
    return out


def _belebele(lang: str, q: str, a: str):
    out = []
    for line in Path(_dl("facebook/belebele", f"data/{lang}.jsonl")).read_text().splitlines():
        d = json.loads(line)
        ctx = f"{d['flores_passage'].strip()}\n{q}: {d['question'].strip()}\n{a}:"
        conts = [" " + d[f"mc_answer{i}"].strip() for i in range(1, 5)]
        out.append({"ctxs": [ctx], "conts": conts, "gold": int(d["correct_answer_num"]) - 1})
    return out


def belebele_tr():
    return _belebele("tur_Latn", "Soru", "Cevap")


def belebele_en():
    return _belebele("eng_Latn", "Question", "Answer")


def _arc(rows, q: str, a: str):
    out = []
    for d in rows:
        ch = _lit(d["choices"])
        if d["answerKey"] not in ch["label"]:
            continue
        out.append({"ctxs": [f"{q}: {d['question'].strip()}\n{a}:"], "conts": [" " + t for t in ch["text"]],
                    "gold": ch["label"].index(d["answerKey"])})
    return out


def arc_tr():
    return _arc(_parquet("malhajar/arc-tr-v0.2", "data/test-00000-of-00001.parquet"), "Soru", "Cevap")


def arc_easy():
    return _arc(_parquet("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet"), "Question", "Answer")


def arc_challenge():
    return _arc(_parquet("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet"), "Question", "Answer")


def hellaswag_tr():
    out = []
    for d in _parquet("malhajar/hellaswag_tr-v0.2", "data/validation-00000-of-00001.parquet"):
        ends = _lit(d["endings"])
        if not d["label"] or len(ends) != 4:
            continue
        out.append({"ctxs": [_hs_pre(d["ctx"])], "conts": [" " + _hs_pre(e) for e in ends], "gold": int(d["label"])})
    return out


def hellaswag_en():
    out = []
    for d in _parquet("Rowan/hellaswag", "data/validation-00000-of-00001.parquet"):
        ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
        out.append({"ctxs": [_hs_pre(d["activity_label"] + ": " + ctx)],
                    "conts": [" " + _hs_pre(e) for e in _lit(d["endings"])], "gold": int(d["label"])})
    return out


def winogrande():
    out = []
    for d in _parquet("allenai/winogrande", "winogrande_xl/validation-00000-of-00001.parquet"):
        s = d["sentence"]
        i = s.index("_")
        cont = " " + s[i + 1:].strip()
        out.append({"ctxs": [s[:i] + d["option1"], s[:i] + d["option2"]], "conts": [cont, cont],
                    "gold": int(d["answer"]) - 1, "unnormalized": True})
    return out


def turkish_mmlu():
    from huggingface_hub import HfApi

    files = sorted(s.rfilename for s in HfApi().dataset_info("AYueksel/TurkishMMLU").siblings
                   if s.rfilename.startswith("test/") and s.rfilename.endswith(".json"))
    out = []
    for f in files:
        for d in json.loads(Path(_dl("AYueksel/TurkishMMLU", f)).read_text()):
            ch = _lit(d["choices"])
            gold = "ABCDE".index(d["answer"].strip())
            if gold >= len(ch):
                continue
            out.append({"ctxs": [f"Soru: {d['question'].strip()}\nCevap:"], "conts": [" " + str(c).strip() for c in ch],
                        "gold": gold})
    return out


def lambada_en():
    out = []
    for d in _parquet("EleutherAI/lambada_openai", "en/test/en.parquet"):
        ctx, last = d["text"].rsplit(" ", 1)
        out.append({"ctx": ctx, "cont": " " + last})
    return out


def _belebele_passages(lang: str):
    seen, out = set(), []
    for line in Path(_dl("facebook/belebele", f"data/{lang}.jsonl")).read_text().splitlines():
        d = json.loads(line)
        if d["link"] in seen:
            continue
        seen.add(d["link"])
        out.append((d["link"], d["flores_passage"].strip()))
    return out


def flores_tr_bpb():
    """Belebele (FLORES) paragrafları: profesyonel çeviri, temiz Türkçe. İngilizcesiyle paralel."""
    return [{"text": t} for _, t in sorted(_belebele_passages("tur_Latn"))]


def flores_en_bpb():
    return [{"text": t} for _, t in sorted(_belebele_passages("eng_Latn"))]


# --------------------------------------------------------------------------- kayıt

@dataclass
class Task:
    name: str
    kind: str  # "mc" | "greedy" | "bpb"
    loader: Callable[[], list[dict]]
    lang: str
    limit: int | None = None  # varsayılan örnek sayısı (None = hepsi)
    random_baseline: float | None = None
    note: str = ""
    _cache: list | None = field(default=None, repr=False)

    def items(self, limit: int | None = None) -> list[dict]:
        if self._cache is None:
            items = self.loader()
            random.Random(0).shuffle(items)  # sabit alt küme
            self._cache = items
        n = limit if limit is not None else self.limit
        return self._cache[:n] if n else self._cache


TASKS = {t.name: t for t in [
    # Türkçe
    Task("xcopa_tr", "mc", xcopa_tr, "tr", None, 0.5, "sağduyu, neden-sonuç (yerel çeviri)"),
    Task("belebele_tr", "mc", belebele_tr, "tr", None, 0.25, "okuduğunu anlama"),
    Task("arc_tr", "mc", arc_tr, "tr", None, 0.25, "fen soruları (çeviri)"),
    Task("hellaswag_tr", "mc", hellaswag_tr, "tr", 1000, 0.25, "cümle tamamlama (çeviri)"),
    Task("turkish_mmlu", "mc", turkish_mmlu, "tr", None, 0.2, "Türk sınav soruları (özgün)"),
    Task("flores_tr_bpb", "bpb", flores_tr_bpb, "tr", None, None, "bit/bayt, temiz Türkçe (düşük iyi)"),
    # İngilizce
    Task("hellaswag_en", "mc", hellaswag_en, "en", 1000, 0.25, "cümle tamamlama"),
    Task("arc_easy", "mc", arc_easy, "en", 1000, 0.25, "fen soruları"),
    Task("arc_challenge", "mc", arc_challenge, "en", None, 0.25, "zor fen soruları"),
    Task("winogrande", "mc", winogrande, "en", None, 0.5, "zamir çözümleme"),
    Task("belebele_en", "mc", belebele_en, "en", None, 0.25, "okuduğunu anlama (belebele_tr ile paralel)"),
    Task("lambada_en", "greedy", lambada_en, "en", 1000, 0.0, "son kelimeyi tahmin"),
    Task("flores_en_bpb", "bpb", flores_en_bpb, "en", None, None, "bit/bayt (flores_tr ile paralel)"),
]}

DEFAULT_TASKS = list(TASKS)
