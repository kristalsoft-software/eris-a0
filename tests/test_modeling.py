"""Model kodunun temel testleri (küçük rastgele model, internet gerekmez).

  python -m pytest tests

Copyright 2026 Kristalsoft (Eren Koçakgöl). Licensed under the Apache License, Version 2.0.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modeling"))

from configuration_eris import ErisConfig  # noqa: E402
from modeling_eris import ErisForCausalLM, ErisRepetitionGuard  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = ErisConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, local_window=16)
    m = ErisForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1) if p.ndim > 1 else p.uniform_(0.5, 1.5)
    return m


def _ids(n, seed):
    return torch.randint(4, 256, (1, n), generator=torch.Generator().manual_seed(seed))


def test_kv_cache_matches_full_forward(model):
    x = _ids(60, 1)  # > yerel pencere (16)
    with torch.no_grad():
        full = model(x, use_cache=False).logits
        out = model(x[:, :20], use_cache=True)
        cache, steps = out.past_key_values, []
        for t in range(20, 60):
            o = model(x[:, t:t + 1], past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            steps.append(o.logits[:, -1])
    assert torch.allclose(full[:, 20:], torch.stack(steps, 1), atol=1e-4)


def test_left_padding(model):
    a, b = _ids(30, 2), _ids(18, 3)
    x = torch.cat([a, torch.cat([torch.full((1, 12), 1), b], 1)], 0)
    mask = torch.cat([torch.ones(1, 30), torch.cat([torch.zeros(1, 12), torch.ones(1, 18)], 1)], 0).long()
    with torch.no_grad():
        batched = model(x, attention_mask=mask).logits
        single = model(b).logits
    assert torch.isfinite(batched).all()
    assert torch.allclose(batched[1:, 12:], single, atol=1e-4)


def test_local_window_limits_context(model):
    """Sadece yerel katmanlar olsaydı pencere dışı token etkisizdi; global katman var, o yüzden etkili olmalı."""
    x = _ids(40, 4)
    y = x.clone()
    y[0, 0] = (y[0, 0] + 1) % 256 + 4 if y[0, 0] < 250 else 5
    with torch.no_grad():
        a, b = model(x).logits[0, -1], model(y).logits[0, -1]
    assert not torch.allclose(a, b)  # global katman uzak tokenı görür


def test_repetition_guard_blocks_ngrams():
    g = ErisRepetitionGuard(prompt_len=2, penalty=1.0, ngram=3)
    ids = torch.tensor([[9, 9, 5, 6, 7, 5, 6]])  # üretilen: 5 6 7 5 6 -> "5 6" ardından 7 yasak
    scores = torch.zeros(1, 10)
    out = g(ids, scores)
    assert out[0, 7] == -float("inf")
    assert torch.isfinite(out[0, [0, 1, 2, 3, 4, 5, 6, 8, 9]]).all()


def test_generate_runs(model):
    out = model.generate(_ids(5, 5), max_new_tokens=6, do_sample=False, eos_token_id=None, pad_token_id=1)
    assert out.shape == (1, 11)
