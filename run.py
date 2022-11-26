from transformers import LayoutLMv2Processor, set_seed
from PIL import Image
import torch
from datasets import load_dataset

from modeling import MemorizableLayoutLMv2Model

set_seed(88)

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", use_fast=False)

def run_inference(model):
    dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
    image_path = dataset["test"][0]["file"]
    image = Image.open(image_path).convert("RGB")

    inputs = [image, image]
    bsz = len(inputs)
    encoding = processor(inputs, return_tensors="pt")

    print(encoding["input_ids"].shape)

    with torch.no_grad():
        outputs = model(**encoding)
    last_hidden_states = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    mems = outputs.mems

    print(last_hidden_states.size())
    print(pooler_output.shape)
    print(mems[0].shape)

    assert last_hidden_states.shape == torch.Size([bsz, 342, 768])
    assert pooler_output.shape == torch.Size([bsz, 768])
    assert len(mems) == model.config.num_hidden_layers
    assert mems[0].shape == torch.Size([bsz, model.mem_len, 768])

def run_training(model):
    dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
    image_path = dataset["test"][0]["file"]
    image = Image.open(image_path).convert("RGB")

    encoding = processor(image, return_tensors="pt")

    mems = None

    model.train()
    for i in range(3):
        outputs = model(**encoding, mems=mems)
        mems = outputs.mems
        print(mems[0].shape)

    outputs.last_hidden_state.sum().backward()
    
    last_hidden_states = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    mems = outputs.mems
    print(last_hidden_states.size())
    print(pooler_output.shape)
    print(mems[0].shape)


if __name__ == "__main__":
    mem_model = MemorizableLayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")
    run_inference(mem_model)
    run_training(mem_model)