"""Değerlendirilen modeller: hepsi Hugging Face transformers üzerinden, aynı arayüzle yüklenir.

Eris modelleri `trust_remote_code=True` ile yüklenir (mimari kodu model deposunda, Apache-2.0).
Puanlama scoring.py'de, tüm modeller için aynı kodla yapılır.

Copyright 2026 Kristalsoft (Eren Koçakgöl). Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelSpec:
    key: str
    hf_id: str
    params_m: float | None = None  # bilgi için (milyon parametre)
    note: str = ""
    tr_made: bool = False  # Türk yapımı
    remote_code: bool = False


# Kimlikler ve boyutlar Hugging Face'te doğrulandı (2026-11, safetensors parametre sayısı).
# 1B altındaki Türk yapımı nedensel (causal) modellerin hepsi burada. Bilerek dışarıda bırakılanlar:
#   TURNA (1.14B, T5 kodlayıcı-kod çözücü), VBART (mBART seq2seq, CC-BY-NC), hamza-xl (1.31B),
#   kanarya-2b (2.06B), Llama-3.2-1B (1.24B), cenkersisman/gpt2-turkish-900m (adına rağmen 124M).
MODELS = {m.key: m for m in [
    ModelSpec("eris-a0-base", "kristalsoft/Eris-A0-Base", 338, "Eris A0 (ön eğitimli)", True, True),
    ModelSpec("eris-a0-instruct", "kristalsoft/Eris-A0-Instruct", 338, "Eris A0 (SFT)", True, True),
    # Türk yapımı, 1B altı
    ModelSpec("tr-gpt2", "ytu-ce-cosmos/turkish-gpt2", 124, "YTÜ COSMOS, Türkçe GPT-2", True),
    ModelSpec("tr-gpt2-medium", "ytu-ce-cosmos/turkish-gpt2-medium", 355, "YTÜ COSMOS, Türkçe GPT-2", True),
    ModelSpec("tr-gpt2-large", "ytu-ce-cosmos/turkish-gpt2-large", 774, "YTÜ COSMOS, Türkçe GPT-2", True),
    ModelSpec("kanarya-750m", "asafaya/kanarya-750m", 738, "Türkçe GPT-J", True),
    ModelSpec("hamza-small", "emrecanacikgoz/hamza-small", 124, "Hamza, Türkçe GPT-2", True),
    ModelSpec("hamza-large", "emrecanacikgoz/hamza-large", 774, "Hamza, Türkçe GPT-2", True),
    # Çok dilli / yabancı, 1B altı
    ModelSpec("smollm2-360m", "HuggingFaceTB/SmolLM2-360M", 362, "İngilizce ağırlıklı, 4T token"),
    ModelSpec("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B", 494, "çok dilli, 18T token"),
    ModelSpec("qwen3-0.6b", "Qwen/Qwen3-0.6B-Base", 596, "çok dilli, 36T token"),
    ModelSpec("xglm-564m", "facebook/xglm-564M", 564, "çok dilli (Türkçe dahil)"),
    ModelSpec("gemma3-270m", "google/gemma-3-270m", 268, "çok dilli. Erişim izni gerekir (HF'de lisansı kabul et)"),
    ModelSpec("gemma3-1b", "google/gemma-3-1b-pt", 1000, "çok dilli. Erişim izni gerekir (HF'de lisansı kabul et)"),
    # Büyük Türkçe referans (1B üstü)
    ModelSpec("kumru-2b", "vngrs/Kumru-2B-Base", 2375, "VNGRS, sıfırdan Türkçe, 300B token", True),
]}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LM:
    """Puanlamanın beklediği arayüz.

    prefix_id: her dizinin başına konan token ya da None. Sadece modelin kendi alışkanlığı: tokenizer
    varsayılan olarak BOS ekliyorsa BOS, eklemiyorsa hiçbir şey. (Ör. Türkçe GPT-2, başta <|endoftext|>
    görünce bozuluyor.) Eris eğitimde belge başı tokenını görmediği için onda da None.
    """

    name: str
    vocab_size: int
    prefix_id: int | None
    max_len: int
    device: str
    n_params: int

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    @torch.no_grad()
    def logits(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class HFLM(LM):
    def __init__(self, hf_id: str, device: str, max_len: int = 2048, name: str | None = None,
                 dtype: torch.dtype | None = None, remote_code: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype or torch.float32,
                                                          trust_remote_code=remote_code).to(device).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        if self.n_params > 1.2e9 and device != "cpu" and dtype is None:
            # büyük modeller için bellek. T4 gibi bf16 desteklemeyen GPU'larda fp16
            half = torch.bfloat16 if device != "cuda" or torch.cuda.is_bf16_supported() else torch.float16
            self.model = self.model.to(half)
        bos = self.tok.bos_token_id
        self.prefix_id = bos if bos is not None and self.tok("a").input_ids[:1] == [bos] else None
        cfg_max = getattr(self.model.config, "max_position_embeddings", None) or getattr(
            self.model.config, "n_positions", None) or max_len
        self.max_len = min(max_len, cfg_max)
        self.device = device
        self.vocab_size = self.model.get_input_embeddings().weight.shape[0]
        self.name = name or hf_id
        self.info = {"hf_id": hf_id}

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    @torch.no_grad()
    def logits(self, ids, attn):
        return self.model(input_ids=ids, attention_mask=attn).logits


def load(key: str, device: str, path: str | None = None) -> LM:
    """Kayıtlı anahtar (eris-a0-base, kumru-2b ...) ya da doğrudan HF kimliği / yerel klasör.

    path: kayıtlı modeli Hub yerine yerel bir klasörden yüklemek için (ör. kendi ince ayarınız).
    """
    spec = MODELS.get(key)
    if spec is None:
        return HFLM(path or key, device, name=key, remote_code=True)
    return HFLM(path or spec.hf_id, device, name=key, remote_code=spec.remote_code)
