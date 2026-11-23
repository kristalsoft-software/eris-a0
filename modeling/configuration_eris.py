"""Eris A0 configuration for Hugging Face transformers.

Copyright 2026 Kristalsoft (Eren Koçakgöl). Licensed under the Apache License, Version 2.0.
"""

from transformers import PretrainedConfig


class ErisConfig(PretrainedConfig):
    """Configuration of the Eris decoder-only language model.

    Architecture summary (pre-norm decoder):
      * grouped-query attention with per-head QK RMSNorm and a per-head sigmoid output gate
      * hybrid attention: every ``global_every``-th layer is global, the others use a local
        causal window of ``local_window`` tokens
      * RoPE with a separate base for local (``rope_theta_local``) and global
        (``rope_theta_global``) layers
      * SwiGLU feed-forward, RMSNorm, tied input/output embeddings, tanh logit soft-capping
    """

    model_type = "eris"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 4096,
        local_window: int = 1024,
        global_every: int = 4,
        rope_theta_local: float = 10_000.0,
        rope_theta_global: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
        final_logit_softcapping: float = 30.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        pad_token_id: int = 1,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.local_window = local_window
        self.global_every = global_every
        self.rope_theta_local = rope_theta_local
        self.rope_theta_global = rope_theta_global
        self.rms_norm_eps = rms_norm_eps
        self.final_logit_softcapping = final_logit_softcapping
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def is_global(self, layer_idx: int) -> bool:
        return layer_idx % self.global_every == self.global_every - 1
