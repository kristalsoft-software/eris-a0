"""Log-olasılık tabanlı puanlama (tüm modeller için aynı kod)."""

from __future__ import annotations

import math
import time

import torch

from .models import LM
from .tasks import Task


def _encode_pair(lm: LM, ctx: str, cont: str) -> tuple[list[int], list[int]]:
    """lm-eval yöntemi: bağlamın sonundaki boşluk devama taşınır, bütün birlikte tokenize edilip bölünür
    (sınırda birleşen tokenlar doğru ayrılsın diye)."""
    n_sp = len(ctx) - len(ctx.rstrip())
    if n_sp:
        ctx, cont = ctx[:-n_sp], ctx[-n_sp:] + cont
    whole = lm.encode(ctx + cont)
    c = lm.encode(ctx)
    pre = [lm.prefix_id] if lm.prefix_id is not None else []
    if whole[: len(c)] != c:  # sınırda birleşme: ayrı tokenize et
        return pre + c, lm.encode(cont)
    return pre + c, whole[len(c):]


@torch.no_grad()
def _score(lm: LM, reqs: list[tuple[list[int], list[int]]], token_budget: int = 2048):
    """Her (bağlam, devam) için (devamın toplam log-olasılığı, devam tokenlarının hepsi en olası mı)."""
    out: list[tuple[float, bool] | None] = [None] * len(reqs)
    seqs = []
    for i, (c, x) in enumerate(reqs):
        s = (c + x)[-(lm.max_len + 1):]  # çok uzunsa baştan kırp (devam korunur)
        seqs.append((i, s, min(len(x), len(s) - 1)))
    seqs.sort(key=lambda t: -len(t[1]))
    # Logit tensörü (batch × T × sözlük) belleği belirler: büyük sözlüklü modellerde batch küçülür.
    token_budget = max(256, int(token_budget * 65536 / getattr(lm, "vocab_size", 65536)))
    j = 0
    while j < len(seqs):
        T = len(seqs[j][1]) - 1
        bs = max(1, token_budget // max(T, 1))
        batch = seqs[j: j + bs]
        j += bs
        ids = torch.zeros((len(batch), T), dtype=torch.long)
        attn = torch.zeros((len(batch), T), dtype=torch.long)
        for r, (_, s, _) in enumerate(batch):
            ids[r, : len(s) - 1] = torch.tensor(s[:-1])
            attn[r, : len(s) - 1] = 1
        logits = lm.logits(ids.to(lm.device), attn.to(lm.device))
        for r, (i, s, n) in enumerate(batch):
            L = len(s) - 1
            lp = torch.log_softmax(logits[r, L - n: L].float(), dim=-1)
            tgt = torch.tensor(s[L - n + 1: L + 1], device=lp.device)
            tok_lp = lp.gather(-1, tgt[:, None])[:, 0]
            greedy = bool((lp.argmax(-1) == tgt).all())
            out[i] = (float(tok_lp.sum()), greedy)
    return out


def _bpb_reqs(lm: LM, text: str):
    """(istekler, puanlanmayan bayt). Önek tokenı olmayan modellerde ilk token puanlanamaz: onun baytları
    paydadan da düşülür. Uzun metinlerde her parça, önündeki en fazla max_len/2 tokenla puanlanır."""
    ids = lm.encode(text)
    if lm.prefix_id is not None:
        full, skipped = [lm.prefix_id] + ids, 0
    else:
        full, skipped = ids, len(lm.decode(ids[:1]).encode("utf-8"))
    if len(full) <= lm.max_len + 1:
        return [(full[:1], full[1:])], skipped
    step = lm.max_len // 2
    return [(full[max(0, s - step): s], full[s: s + step]) for s in range(1, len(full), step)], skipped


def run_task(lm: LM, task: Task, limit: int | None = None) -> dict:
    items = task.items(limit)
    t0 = time.time()
    if task.kind == "mc":
        reqs, index, conts = [], [], []
        for k, it in enumerate(items):
            ctxs = it["ctxs"] * len(it["conts"]) if len(it["ctxs"]) == 1 else it["ctxs"]
            for ctx, cont in zip(ctxs, it["conts"]):
                reqs.append(_encode_pair(lm, ctx, cont))
                index.append(k)
                conts.append(cont)
        scores = _score(lm, reqs)
        per = [[] for _ in items]
        for k, (lp, _), cont in zip(index, scores, conts):
            per[k].append((lp, lp / max(1, len(cont))))  # acc_norm: karakter başına (lm-eval gibi)
        acc = sum(max(range(len(p)), key=lambda i: p[i][0]) == it["gold"] for p, it in zip(per, items))
        acc_norm = sum(max(range(len(p)), key=lambda i: p[i][1]) == it["gold"] for p, it in zip(per, items))
        n = len(items)
        res = {"acc": acc / n, "acc_norm": acc_norm / n, "n": n}
        # Ana ölçü: seçenek uzunlukları farklı görevlerde uzunluğa göre normalize (lm-eval acc_norm),
        # winogrande'de devam metni aynı olduğu için ham acc.
        res["main"] = res["acc"] if items and items[0].get("unnormalized") else res["acc_norm"]
    elif task.kind == "greedy":
        reqs = [_encode_pair(lm, it["ctx"], it["cont"]) for it in items]
        scores = _score(lm, reqs)
        acc = sum(g for _, g in scores) / len(items)
        ppl = math.exp(-sum(lp for lp, _ in scores) / len(items))
        res = {"acc": acc, "ppl": ppl, "n": len(items), "main": acc}
    elif task.kind == "bpb":
        reqs, nbytes = [], 0
        for it in items:
            r, skipped = _bpb_reqs(lm, it["text"])
            reqs += r
            nbytes += len(it["text"].encode("utf-8")) - skipped
        scores = _score(lm, reqs)
        nll = -sum(lp for lp, _ in scores)
        ntok = sum(len(x) for _, x in reqs)
        res = {"bpb": nll / math.log(2) / nbytes, "bytes_per_token": nbytes / ntok, "n": len(items)}
        res["main"] = res["bpb"]
    else:
        raise ValueError(task.kind)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def fertility(lm: LM, texts: list[str]) -> float:
    """Kelime başına token (düşük = tokenizer o dilde verimli)."""
    words = sum(len(t.split()) for t in texts)
    return sum(len(lm.encode(t)) for t in texts) / max(1, words)
