import re

# from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import time
from PIL import Image

processor = DonutProcessor.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2", use_fast=True
)
model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2", dtype="float16"
)

device = "cuda"
model.to(device)  # doctest: +IGNORE_RESULT
# model.encoder.compile()
# model.decoder.compile()

# # load document image
# dataset = load_dataset("hf-internal-testing/example-documents", split="test")
# image = dataset[2]["image"]

image = Image.open("./example.png")

# prepare decoder inputs + image preprocessing
prep_start = time.perf_counter()
# task_prompt = (
#     "<s_docvqa><s_question>What time is the coffee break?</s_question><s_answer>"
# )
task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(
    task_prompt, add_special_tokens=False, return_tensors="pt"
).input_ids
pixel_values = processor(image, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(torch.float16)
prep_time = time.perf_counter() - prep_start
print(f"Preprocess time (s)={prep_time:.4f}")

for _ in range(10):
    start_time = time.perf_counter()
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=128,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=None,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    print("Time to generate:", time.perf_counter() - start_time)


print(outputs.sequences)
sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
    processor.tokenizer.pad_token, ""
)
sequence = re.sub(
    r"<.*?>", "", sequence, count=1
).strip()  # remove first task start token
print(processor.token2json(sequence))
