"""Eris A0 model for Hugging Face transformers (PyTorch, inference and fine-tuning).

Copyright 2026 Kristalsoft (Eren Koçakgöl). Licensed under the Apache License, Version 2.0.

Load with:
    AutoModelForCausalLM.from_pretrained("kristalsoft/<repo>", trust_remote_code=True)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, LogitsProcessor, LogitsProcessorList, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:  # relative import when loaded as remote code, plain import otherwise
    from .configuration_eris import ErisConfig
except ImportError:  # pragma: no cover
    from configuration_eris import ErisConfig


class ErisRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def _rope_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D); cos/sin: (B, 1, T, D/2). Half-split rotation."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class ErisAttention(nn.Module):
    def __init__(self, config: ErisConfig, layer_idx: int):
        super().__init__()
        d, hd = config.hidden_size, config.head_dim
        self.layer_idx = layer_idx
        self.is_global = config.is_global(layer_idx)
        self.n_head, self.n_kv, self.hd = config.num_attention_heads, config.num_key_value_heads, hd
        self.q_proj = nn.Linear(d, self.n_head * hd, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * hd, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * hd, bias=False)
        self.o_proj = nn.Linear(self.n_head * hd, d, bias=False)
        self.gate_proj = nn.Linear(d, self.n_head, bias=False)  # per-head sigmoid output gate
        self.q_norm = ErisRMSNorm(hd, config.rms_norm_eps)
        self.k_norm = ErisRMSNorm(hd, config.rms_norm_eps)

    def forward(self, x, rope, mask, past_key_values):
        B, T, _ = x.shape
        cos, sin = rope
        q = self.q_proj(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        rep = self.n_head // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (B, H, T, hd)
        g = torch.sigmoid(self.gate_proj(x)).transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)
        y = (y * g).transpose(1, 2).reshape(B, T, self.n_head * self.hd)
        return self.o_proj(y)


class ErisMLP(nn.Module):
    def __init__(self, config: ErisConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ErisDecoderLayer(nn.Module):
    def __init__(self, config: ErisConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = ErisRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = ErisAttention(config, layer_idx)
        self.post_attention_layernorm = ErisRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = ErisMLP(config)

    def forward(self, x, rope, mask, past_key_values):
        x = x + self.self_attn(self.input_layernorm(x), rope, mask, past_key_values)
        return x + self.mlp(self.post_attention_layernorm(x))


class ErisPreTrainedModel(PreTrainedModel):
    config_class = ErisConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["ErisDecoderLayer"]
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, ErisRMSNorm):
            nn.init.ones_(module.weight)


class ErisModel(ErisPreTrainedModel):
    def __init__(self, config: ErisConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=None)
        self.layers = nn.ModuleList([ErisDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = ErisRMSNorm(config.hidden_size, config.rms_norm_eps)
        # RoPE frequencies are computed in forward (no buffers): robust to meta-device loading.
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _masks(self, attention_mask, q_pos, kv_len, B, device):
        """Boolean masks (B, 1, T, S): causal + key padding, local layers also windowed."""
        k_pos = torch.arange(kv_len, device=device)
        causal = k_pos[None, :] <= q_pos[:, None]  # (T, S)
        allowed = causal[None, None]
        if attention_mask is not None:
            keep = attention_mask[:, None, None, -kv_len:].bool()  # (B, 1, 1, S)
            allowed = allowed & keep
            # a query always sees itself, so fully padded rows stay finite
            allowed = allowed | (k_pos[None, :] == q_pos[:, None])[None, None]
        allowed = allowed.expand(B, 1, -1, -1)
        window = (q_pos[:, None] - k_pos[None, :]) < self.config.local_window
        return {"global": allowed, "local": allowed & window[None, None]}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values: Cache | None = None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        B, T, _ = h.shape
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        q_pos = torch.arange(past, past + T, device=h.device)
        if position_ids is None:
            position_ids = q_pos[None].expand(B, -1)
        masks = self._masks(attention_mask, q_pos, past + T, B, h.device)

        ropes = {}
        hd = self.config.head_dim
        for kind, base in (("local", self.config.rope_theta_local), ("global", self.config.rope_theta_global)):
            inv = _rope_inv_freq(hd, base).to(h.device)
            freqs = position_ids[:, :, None].float() * inv[None, None, :]  # (B, T, D/2)
            ropes[kind] = (freqs.cos()[:, None], freqs.sin()[:, None])

        for layer in self.layers:
            kind = "global" if layer.self_attn.is_global else "local"
            h = layer(h, ropes[kind], masks[kind], past_key_values if use_cache else None)
        return BaseModelOutputWithPast(last_hidden_state=self.norm(h),
                                       past_key_values=past_key_values if use_cache else None)


class ErisRepetitionGuard(LogitsProcessor):
    """Repetition penalty + no-repeat n-gram applied to the *generated* tokens only.

    The built-in ``repetition_penalty`` / ``no_repeat_ngram_size`` also act on the prompt, which pushes a small
    model away from words of the question (e.g. "başkenti"). This mirrors how Eris was evaluated.
    """

    def __init__(self, prompt_len: int, penalty: float = 1.15, ngram: int = 4):
        self.start, self.penalty, self.n = prompt_len, penalty, ngram

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            gen = input_ids[b, self.start:].tolist()
            if gen and self.penalty != 1.0:
                idx = torch.tensor(sorted(set(gen)), device=scores.device)
                s = scores[b, idx]
                scores[b, idx] = torch.where(s > 0, s / self.penalty, s * self.penalty)
            if self.n and len(gen) >= self.n - 1:
                prefix = tuple(gen[len(gen) - self.n + 1:]) if self.n > 1 else ()
                for i in range(len(gen) - self.n + 1):
                    if tuple(gen[i:i + self.n - 1]) == prefix:
                        scores[b, gen[i + self.n - 1]] = -float("inf")
        return scores


class ErisForCausalLM(ErisPreTrainedModel, GenerationMixin):
    """Eris causal LM. The output projection is tied to the input embedding matrix."""

    @torch.no_grad()
    def chat(self, tokenizer, messages, max_new_tokens: int = 256, temperature: float = 0.3, top_k: int = 40,
             repetition_penalty: float = 1.15, no_repeat_ngram: int = 4, **generate_kwargs) -> str:
        """Answer a chat. ``messages`` is a string or a list of {"role", "content"} dicts."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True).to(self.device)
        n = enc["input_ids"].shape[1]
        guard = LogitsProcessorList([ErisRepetitionGuard(n, repetition_penalty, no_repeat_ngram)])
        out = self.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                            temperature=temperature if temperature > 0 else None,
                            top_k=top_k if temperature > 0 else None, repetition_penalty=1.0,
                            no_repeat_ngram_size=0, logits_processor=guard, **generate_kwargs)
        return tokenizer.decode(out[0, n:], skip_special_tokens=True).strip()

    def __init__(self, config: ErisConfig):
        super().__init__(config)
        self.model = ErisModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return None  # logits use the (tied) input embedding matrix directly

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values: Cache | None = None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep: int = 0,
        **kwargs,
    ):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache)
        h = out.last_hidden_state
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        logits = F.linear(h, self.model.embed_tokens.weight).float()
        c = self.config.final_logit_softcapping
        if c:
            logits = c * torch.tanh(logits / c)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                                   ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=out.past_key_values)


__all__ = ["ErisForCausalLM", "ErisModel", "ErisPreTrainedModel", "ErisRepetitionGuard"]
