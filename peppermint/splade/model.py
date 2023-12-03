from transformers import AutoModelForMaskedLM, AutoTokenizer

import torch
import torch.nn.functional as F


def calculate_terms(
        logits: torch.Tensor,
        tokenizer: AutoTokenizer
) -> list[str]:
    weights = torch.log(1 + F.relu(torch.exp(logits))).sum(1)
    weights = (weights > 1.0) * weights
    
    ixs = torch.topk(weights.squeeze(), 100).indices

    terms = [tokenizer.decode([w.item()]) for w in ixs]

    return terms


if __name__ == "__main__":
    model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    text = "We hold these truths to be self-evident that all men are created equal."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input).logits

    print(calculate_terms(output, tokenizer))