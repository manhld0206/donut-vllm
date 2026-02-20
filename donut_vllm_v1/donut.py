# SPDX-License-Identifier: Apache-2.0
# Copyright: Donut vLLM integration

from __future__ import annotations
from vllm.compilation.decorators import support_torch_compile
from typing import Literal, Annotated, Optional, Union

import math
from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import nn
from transformers import MBartConfig, VisionEncoderDecoderConfig
from transformers.models.swin.modeling_swin import (
    window_partition,
    window_reverse,
    SwinConfig,
)

from vllm.config import CacheConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.swin import (
    SwinEmbeddings,
    SwinLayer as VllmSwinLayer,
    SwinModel,
    SwinPatchMerging,
)
from vllm.model_executor.models.utils import maybe_prefix, flatten_bn, AutoWeightsLoader
from vllm.model_executor.models.whisper import WhisperAttention, WhisperCrossAttention
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from functools import lru_cache

from vllm.utils.tensor_schema import TensorSchema, TensorShape
from transformers import AutoTokenizer
from transformers.models.donut.image_processing_donut_fast import (
    DonutImageProcessorFast,
)
from transformers.models.donut.processing_donut import DonutProcessor
from vllm.v1.attention.backend import AttentionType


class DonutImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - c: Number of channels (3)
        - h: Height
        - w: Width
    """

    type: Literal["pixel_values"]
    data: Annotated[torch.Tensor, TensorShape("b", 3, "h", "w")]


class DonutProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs):
        return self.ctx.get_hf_processor(**kwargs)

    @property
    def skip_prompt_length_check(self) -> bool:
        return True  # Because the encoder prompt is padded

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 1}

    def get_num_image_tokens(self) -> int:
        config = self.get_hf_config()
        encoder_config: SwinConfig = config.encoder

        image_size = encoder_config.image_size
        if isinstance(image_size, (list, tuple)):
            h, w = image_size
        else:
            h = w = image_size
        patch_size = encoder_config.patch_size

        h_patches = h // patch_size
        w_patches = w // patch_size

        depths = encoder_config.depths
        num_merges = len(depths) - 1

        final_h = h_patches // (2**num_merges)
        final_w = w_patches // (2**num_merges)

        return final_h * final_w


class DonutDummyInputsBuilder(BaseDummyInputsBuilder[DonutProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        config = self.info.get_hf_config().encoder
        image_size = config.image_size
        if isinstance(image_size, (list, tuple)):
            height, width = image_size
        else:
            height = width = image_size

        num_images = mm_counts.get("image", 0)
        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=width,
                height=height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class DonutMultiModalProcessor(EncDecMultiModalProcessor[DonutProcessingInfo]):
    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
    ) -> str | list[int]:
        return [0]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        if mm_data:
            processed_outputs = super()._call_hf_processor(
                prompt, mm_data, mm_kwargs, tok_kwargs
            )
        else:
            hf_processor = self.info.get_hf_processor()
            tokenizer = hf_processor.tokenizer
            processed_outputs = tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            )
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"))

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        image_tokens = self.info.get_num_image_tokens()
        return {"image": image_tokens}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        num_image_tokens = self.info.get_num_image_tokens()
        return [
            PromptReplacement(
                modality="image",
                target=[0],
                replacement=[0] * num_image_tokens,
            )
        ]


class DonutSwinModel(SwinModel):
    pass


class MBartLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(position_ids + self.offset)


class MBartScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


class MBartDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MBartConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            attn_type=AttentionType.DECODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.encoder_attn = WhisperCrossAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        ffn_hidden_size = self.embed_dim
        ffn_intermediate_size = config.decoder_ffn_dim
        self.fc1 = ColumnParallelLinear(
            ffn_hidden_size,
            ffn_intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            ffn_intermediate_size,
            ffn_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.activation_fn = get_act_fn(config.activation_function)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = decoder_hidden_states
        hidden_states = self.self_attn_layer_norm(decoder_hidden_states)

        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)
        hidden_states, _ = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@support_torch_compile(dynamic_arg_dims={"decoder_input_ids": 0, "positions": -1})
class MBartDecoder(nn.Module):
    def __init__(
        self,
        *,
        config: MBartConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = MBartScaledWordEmbedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
            embed_scale=embed_scale,
        )
        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.layers = nn.ModuleList(
            [
                MBartDecoderLayer(
                    config,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.decoder_layers)
            ]
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_input_ids(decoder_input_ids)

        position_embeds = self.embed_positions(positions)
        hidden_states = self.layernorm_embedding(inputs_embeds + position_embeds)

        for layer in self.layers:
            hidden_states = layer(
                decoder_hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


@MULTIMODAL_REGISTRY.register_processor(
    DonutMultiModalProcessor,
    info=DonutProcessingInfo,
    dummy_inputs=DonutDummyInputsBuilder,
)
class DonutForConditionalGeneration(nn.Module, SupportsMultiModal):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        if not isinstance(config, VisionEncoderDecoderConfig):
            raise TypeError(
                "DonutForConditionalGeneration expects a VisionEncoderDecoderConfig."
            )

        encoder_type = getattr(config.encoder, "model_type", "")
        decoder_type = getattr(config.decoder, "model_type", "")
        if encoder_type not in ("donut-swin", "swin") or decoder_type != "mbart":
            raise ValueError(
                "DonutForConditionalGeneration only supports "
                "donut-swin encoder with mbart decoder."
            )

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.encoder_config = config.encoder
        self.decoder_config = config.decoder

        with self._mark_tower_model(vllm_config, "image"):
            self.encoder = DonutSwinModel(
                self.encoder_config,
                quant_config=quant_config,
                prefix=f"{prefix}.encoder",
            )

        with self._mark_language_model(vllm_config):
            self.decoder = MBartDecoder(
                config=self.decoder_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.decoder",
            )

        self.vocab_size = self.decoder_config.vocab_size
        self.lm_head = ParallelLMHead(
            self.vocab_size,
            self.decoder_config.d_model,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.vocab_size)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")

    @classmethod
    def get_generation_prompt(

    ):
        pass

    def _parse_and_validate_image_input(self, **kwargs: object):
        pixel_values: Optional[
            Union[list[list[torch.Tensor]], list[torch.Tensor], torch.Tensor]
        ] = kwargs.pop("pixel_values", None)
        image_embeds: Optional[
            Union[list[list[torch.Tensor]], list[torch.Tensor], torch.Tensor]
        ] = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None and image_embeds is not None:
            raise ValueError("Both pixel values and image embeds are provided.")

        if pixel_values is not None:
            h, w = self.config.encoder.image_size
            return DonutImagePixelInputs(
                type="pixel_values",
                data=pixel_values,
                resolve_bindings={
                    "h": h,
                    "w": w,
                },
            )

        if image_embeds is not None:
            raise NotImplementedError

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(self, image_input: DonutImagePixelInputs) -> torch.Tensor:
        # assert image_input["type"] == "pixel_values"
        pixel_values = image_input["data"]
        dtype = next(self.encoder.parameters()).dtype
        pixel_values = pixel_values.to(dtype)
        return self.encoder(pixel_values)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        pixel_values = self._parse_and_validate_image_input(**kwargs)
        if pixel_values is None:
            return None
        vision_embeddings = self._process_image_input(pixel_values)
        return vision_embeddings.unbind(dim=0)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        return self.decoder.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        inputs_embeds = None
        if encoder_outputs:
            inputs_embeds = torch.cat(encoder_outputs, dim=0)
        hidden_states = self.decoder(
            decoder_input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: object | None = None,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)


    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loaded: set[str] = set()
        encoder_weights: list[tuple[str, torch.Tensor]] = []
        decoder_weights: list[tuple[str, torch.Tensor]] = []
        lm_head_weights: list[tuple[str, torch.Tensor]] = []
        lm_head_dict = dict(self.lm_head.named_parameters())

        for name, w in weights:
            if name.startswith("encoder."):
                trimmed = name[len("encoder.") :]
                if trimmed.startswith("donut."):
                    trimmed = trimmed[len("donut.") :]
                encoder_weights.append((trimmed, w))
                continue

            if name.startswith("decoder."):
                trimmed = name[len("decoder.") :]
                if trimmed == "final_logits_bias":
                    continue
                if trimmed.startswith("lm_head."):
                    lm_name = trimmed[len("lm_head.") :]
                    if lm_name in lm_head_dict:
                        lm_head_weights.append((lm_name, w))
                    continue
                if trimmed.startswith("model."):
                    trimmed = trimmed[len("model.") :]
                if trimmed.startswith("shared."):
                    trimmed = trimmed.replace("shared.", "embed_tokens.", 1)
                if trimmed.startswith("decoder."):
                    trimmed = trimmed[len("decoder.") :]
                if trimmed.startswith("embed_tokens.") and trimmed not in (
                    "embed_tokens.weight",
                    "embed_tokens.bias",
                ):
                    pass
                decoder_weights.append((trimmed, w))
                continue

            if name.startswith("lm_head."):
                lm_name = name[len("lm_head.") :]
                if lm_name in lm_head_dict:
                    lm_head_weights.append((lm_name, w))
                continue

        if encoder_weights:
            enc_loaded = self.encoder.load_weights(encoder_weights)
            loaded |= {f"encoder.{n}" for n in enc_loaded}
        if decoder_weights:
            dec_loaded = self.decoder.load_weights(decoder_weights)
            loaded |= {f"decoder.{n}" for n in dec_loaded}
        if lm_head_weights:
            loader = AutoWeightsLoader(self.lm_head)
            lm_loaded = loader.load_weights(lm_head_weights)
            loaded |= {f"lm_head.{n}" for n in lm_loaded}
        return loaded
