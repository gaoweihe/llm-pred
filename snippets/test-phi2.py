import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model_name = {
  "phi-2": "microsoft/phi-2",
  "phi-3": "microsoft/Phi-3-mini-4k-instruct"
}
model = AutoModelForCausalLM.from_pretrained(model_name["phi-3"], torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name["phi-3"], trust_remote_code=True)