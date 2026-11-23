import vllm
import torch
import transformers

from transformers import AutoTokenizer

print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B"
)

print("Tokenizer:", type(tokenizer).__name__)

print(
    "Has all_special_tokens_extended:",
    hasattr(tokenizer, "all_special_tokens_extended")
)

assert hasattr(tokenizer, "all_special_tokens_extended")

print("Tokenizer compatibility PASS")