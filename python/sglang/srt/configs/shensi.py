import logging
from typing import List

try:
    from transformers import DeepseekV3Config as _ShensiConfigBase
except ImportError:  # pragma: no cover - transformers without DeepSeek-V3
    from transformers import PretrainedConfig as _ShensiConfigBase

logger = logging.getLogger(__name__)

# HF `layer_types` -> SGLang's per-layer integer `compress_ratios`.
_LAYER_TYPE_TO_COMPRESS_RATIO = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}

# Deferred past `super().__init__` so DeepseekV3Config's strict validator does
# not reject `hash_moe` / `moe` mlp layer types.
_DEFERRED_KEYS = ("layer_types", "mlp_layer_types", "rope_parameters")


class ShensiConfig(_ShensiConfigBase):
    model_type = "shensi"

    def __init__(self, **kwargs):
        deferred = {key: kwargs.pop(key) for key in _DEFERRED_KEYS if key in kwargs}
        super().__init__(**kwargs)
        for key, value in deferred.items():
            setattr(self, key, value)

        self.n_shared_experts = 0

        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.num_attention_heads
            self.head_dim = head_dim

        partial_rotary_factor = getattr(self, "partial_rotary_factor", None)
        if partial_rotary_factor is None:
            partial_rotary_factor = getattr(self, "qk_rope_head_dim", 64) / head_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.qk_rope_head_dim = int(head_dim * partial_rotary_factor)
        self.qk_nope_head_dim = head_dim - self.qk_rope_head_dim
        self.v_head_dim = head_dim

        self.window_size = getattr(self, "sliding_window", 128)

        mlp_layer_types = getattr(self, "mlp_layer_types", None)
        if mlp_layer_types is None:
            n_hash = getattr(self, "default_num_hash_layers", 3)
            mlp_layer_types = ["hash_moe"] * min(self.num_hidden_layers, n_hash) + [
                "moe"
            ] * max(0, self.num_hidden_layers - n_hash)
        self.mlp_layer_types = list(mlp_layer_types[: self.num_hidden_layers])

        self.attn_res_block_size = getattr(self, "attn_res_block_size", 4)

        self.compress_ratios = self._derive_compress_ratios()
        self.num_hash_layers = self._derive_num_hash_layers()

        self._flatten_rope_parameters()

    def attn_res_block_layer_types(self) -> List[str]:
        n_hash = self.mlp_layer_types.count("hash_moe")
        block_size = self.attn_res_block_size
        return [
            "block_write_layer"
            if i == 0 or (i >= n_hash and (i - n_hash) % block_size == 0)
            else "block_read_layer"
            for i in range(self.num_hidden_layers)
        ]

    def _derive_compress_ratios(self) -> List[int]:
        layer_types = getattr(self, "layer_types", None)
        if layer_types is None:
            n = self.num_hidden_layers
            layer_types = ["heavily_compressed_attention"] * min(n, 2) + [
                "compressed_sparse_attention"
                if i % 2
                else "heavily_compressed_attention"
                for i in range(max(n - 2, 0))
            ]
        ratios = []
        for layer_type in layer_types[: self.num_hidden_layers]:
            if layer_type not in _LAYER_TYPE_TO_COMPRESS_RATIO:
                raise ValueError(f"Unknown Shensi layer_type: {layer_type!r}")
            ratios.append(_LAYER_TYPE_TO_COMPRESS_RATIO[layer_type])
        return ratios

    def _derive_num_hash_layers(self) -> int:
        mlp_layer_types = getattr(self, "mlp_layer_types", None)
        if mlp_layer_types is None:
            return getattr(self, "default_num_hash_layers", 3)
        num_hash = 0
        for layer_type in mlp_layer_types:
            if layer_type != "hash_moe":
                break
            num_hash += 1
        return num_hash

    def _flatten_rope_parameters(self) -> None:
        # HF nests the YaRN params under rope_parameters["compress"]; V4 reads them flat.
        rope_parameters = getattr(self, "rope_parameters", None)
        if not isinstance(rope_parameters, dict):
            return
        main = rope_parameters.get("main")
        compress = rope_parameters.get("compress")
        if not isinstance(main, dict) or not isinstance(compress, dict):
            return
        flat = dict(compress)
        flat["rope_theta"] = main.get("rope_theta", getattr(self, "rope_theta", 10000))
        self.rope_parameters = flat


__all__ = ["ShensiConfig"]
