import os
import requests
import torch
import tiktoken
from model_initial import GPTModel, generate_text_simple, GPT_CONFIG_124M
from model_training import text_to_token_ids, token_ids_to_text

file_name = "gpt2-small-124M.pth"
url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"



if not os.path.exists(file_name):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(file_name, "wb") as f:
        f.write(response.content)
    print(f"Downloaded to {file_name}")

GPT_CONFIG_124M.update({"qkv_bias": True}) # Query-Key-Value bias.  # Only change this from False to True
model_gpt = GPTModel(GPT_CONFIG_124M)
model_gpt.load_state_dict(torch.load(file_name, weights_only=True))
model_gpt.eval()

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

inference_device = torch.device("cpu")
model_gpt.to(inference_device)


token_ids = generate_text_simple(
    model=model_gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
)
"""
Using the generate_text_simple function (from the previous chapter) that we used earlier inside the simple training function,
we can generate new text one word (or token) at a time. As explained in section 5.1.2, 
the next generated token is the token corresponding to the largest probability score 
among all tokens in the vocabulary
"""

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))



def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

torch.manual_seed(123)
token_ids = generate(
    model=model_gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=50,      
    temperature=1.5
) # Change top_k and temperature will change the output text.
print("\n")
print("Output text from generate():\n", token_ids_to_text(token_ids, tokenizer))
