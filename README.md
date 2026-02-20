# Overview
- Implementation of Donut [model](https://huggingface.co/docs/transformers/en/model_doc/donut) in vLLM v1 engine
- Donut was originally supported in vLLM v0 engine but dropped in v1
- There are still some multimodal encoder decoder in vLLM v1 now like [whisper](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/whisper/) or [nvidia nemotron parse](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/nemotron_parse/)
- The donut vllm v1 engine implementation can be found in [donut_vllm_v1/donut.py](donut_vllm_v1/donut.py) and took inspiration from 2 above models

# Benchmark
- Set generation config to ignore eos to always generate until max length
- Run with batch size = 1 only
- The benchmarks were run on Nvidia RTX 4070 GPU
- Result

## [Huggingface Donut](test_donut_original.py)

```bash
uv run test_donut_original.py
```
```bash
...
Time to generate: 0.2661869579997074
Time to generate: 0.266351776999727
Time to generate: 0.264271734000431
Time to generate: 0.27392077799959225
```

## [vLLM Donut](test_donut_vllm.py)
```bash
uv run test_donut_vllm.py
```
```bash
...
Time to generate: 0.13058361200000945
Time to generate: 0.13091194400021777
Time to generate: 0.13163644700034638
Time to generate: 0.1315439370000604
```
