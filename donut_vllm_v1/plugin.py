# SPDX-License-Identifier: Apache-2.0


def register() -> None:
    # Register the VisionEncoderDecoderModel architecture for Donut.

    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "VisionEncoderDecoderModel",
        "donut_vllm_v1.donut:DonutForConditionalGeneration",
    )
