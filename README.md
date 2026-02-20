# Overview
- Implementation of Donut [model](https://huggingface.co/docs/transformers/en/model_doc/donut) in vLLM v1 engine
- The donut vllm implementation can be found [here](donut_vllm_v1/donut.py)

# Benchmark
- Set generation config to ignore eos to always generate until max length
- Run with batch size = 1 only
- The benchmarks were run on Nvidia RTX 4070 GPU
- Result

## [Huggingface Donut](test_donut_original.py)

```bash
uv run test_donut_original.py
...
Time to generate: 0.2661869579997074
Time to generate: 0.266351776999727
Time to generate: 0.264271734000431
Time to generate: 0.27392077799959225
```

## [vLLM Donut](test_donut_vllm.py)
```bash
...
Time to generate: 0.13058361200000945
Time to generate: 0.13091194400021777
Time to generate: 0.13163644700034638
Time to generate: 0.1315439370000604
```
