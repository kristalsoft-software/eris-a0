"""Eris (HF formatı) -> tarayıcı için ONNX (KV-cache'li, batch 1), isteğe bağlı int8.

  pip install onnx onnxruntime onnxscript
  python onnx_export/export_onnx.py --hf kristalsoft/Eris-A0-Instruct --out ./eris-a0-onnx

Hazır int8 dosyası model deposunda: kristalsoft/Eris-A0-Instruct, onnx/model_int8.onnx.

Copyright 2026 Kristalsoft (Eren Koçakgöl). Licensed under the Apache License, Version 2.0.

Graf (tek adım):
  girdi:  input_ids (1, T) int64, past_key / past_value (L, KV, P, D) float32
  çıktı:  logits (1, V) son konum için, present_key / present_value (L, KV, P+T, D)
İlk adımda P = 0. Maske, RoPE ve yerel pencere grafın içinde P'den hesaplanır.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ErisStep(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.m = hf_model.model
        c = hf_model.config
        self.c = c
        self.window = c.local_window
        self.softcap = c.final_logit_softcapping
        hd = c.head_dim
        self.register_buffer("inv_local", 1.0 / (c.rope_theta_local ** (torch.arange(0, hd, 2).float() / hd)))
        self.register_buffer("inv_global", 1.0 / (c.rope_theta_global ** (torch.arange(0, hd, 2).float() / hd)))

    @staticmethod
    def _rope(x, cos, sin):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(self, input_ids, past_key, past_value):
        c = self.c
        T = input_ids.shape[1]
        P = past_key.shape[2]
        h = self.m.embed_tokens(input_ids)  # (1, T, d)
        q_pos = torch.arange(T, device=h.device) + P
        k_pos = torch.arange(T + P, device=h.device)
        causal = k_pos[None, :] <= q_pos[:, None]
        local = causal & ((q_pos[:, None] - k_pos[None, :]) < self.window)
        ropes = {}
        for kind, inv in (("local", self.inv_local), ("global", self.inv_global)):
            f = q_pos[:, None].float() * inv[None, :]
            ropes[kind] = (f.cos()[None, None], f.sin()[None, None])  # (1, 1, T, D/2)
        nh, nkv, hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        keys, values = [], []
        for i, layer in enumerate(self.m.layers):
            a = layer.self_attn
            kind = "global" if a.is_global else "local"
            cos, sin = ropes[kind]
            x = layer.input_layernorm(h)
            q = a.q_proj(x).view(1, T, nh, hd).transpose(1, 2)
            k = a.k_proj(x).view(1, T, nkv, hd).transpose(1, 2)
            v = a.v_proj(x).view(1, T, nkv, hd).transpose(1, 2)
            q, k = self._rope(a.q_norm(q), cos, sin), self._rope(a.k_norm(k), cos, sin)
            k = torch.cat([past_key[i:i + 1], k], dim=2)  # (1, nkv, P+T, hd)
            v = torch.cat([past_value[i:i + 1], v], dim=2)
            keys.append(k)
            values.append(v)
            rep = nh // nkv
            kr = k.repeat_interleave(rep, dim=1)
            vr = v.repeat_interleave(rep, dim=1)
            att = (q @ kr.transpose(-1, -2)) / (hd ** 0.5)
            mask = causal if a.is_global else local
            att = att.masked_fill(~mask[None, None], float("-inf")).softmax(-1)
            y = att @ vr  # (1, nh, T, hd)
            g = torch.sigmoid(a.gate_proj(x)).transpose(1, 2).unsqueeze(-1)
            y = (y * g).transpose(1, 2).reshape(1, T, nh * hd)
            h = h + a.o_proj(y)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        hl = self.m.norm(h[:, -1])  # (1, d)
        logits = F.linear(hl, self.m.embed_tokens.weight)
        if self.softcap:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits, torch.cat(keys, 0), torch.cat(values, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="HF kimliği ya da klasör")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-quant", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(args.hf, trust_remote_code=True, dtype=torch.float32).eval()
    step = ErisStep(hf).eval()
    c = hf.config
    L, KV, D = c.num_hidden_layers, c.num_key_value_heads, c.head_dim
    args.out.mkdir(parents=True, exist_ok=True)
    fp32 = args.out / "model_fp32.onnx"
    ids = torch.tensor([[0, 5, 6, 7]])
    pk = torch.zeros(L, KV, 3, D)
    with torch.no_grad():
        torch.onnx.export(
            step, (ids, pk, pk.clone()), str(fp32), dynamo=False, opset_version=17,
            input_names=["input_ids", "past_key", "past_value"],
            output_names=["logits", "present_key", "present_value"],
            dynamic_axes={"input_ids": {1: "T"}, "past_key": {2: "P"}, "past_value": {2: "P"},
                          "present_key": {2: "PT"}, "present_value": {2: "PT"}},
        )
    print("yazıldı:", fp32, f"{fp32.stat().st_size / 1e6:.0f} MB", flush=True)
    if args.no_quant:
        return
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q8 = args.out / "model_int8.onnx"
    quantize_dynamic(str(fp32), str(q8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"],
                     per_channel=True)
    print("yazıldı:", q8, f"{q8.stat().st_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
