import torch
import tiktoken
from model_initial import GPTModel, generate_text_simple
from model_training import text_to_token_ids, GPT_CONFIG_124M, token_ids_to_text

model = GPTModel(GPT_CONFIG_124M)

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

inference_device = torch.device("cpu")
model.to(inference_device)

model.load_state_dict(torch.load("model_trained.pth", map_location=inference_device, weights_only=True))
model.eval()

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device), #You are a genius
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
