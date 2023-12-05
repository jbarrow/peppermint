from transformers import AutoModelForMaskedLM, AutoTokenizer

import torch
import torch.nn.functional as F


class SPLADE:
    def __init__(self, model: str | None = None) -> None:
        model = model or "naver/splade-cocondenser-ensembledistil"
        self.model = AutoModelForMaskedLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.idx2token = {
            idx: token for token, idx in self.tokenizer.get_vocab().items()
        }

    def __call__(self, inputs: list[str]) -> dict[int, float]:
        encoded = self.tokenizer(inputs, return_tensors="pt", padding=True)
        outputs = self.model(**encoded).logits

        vec = torch.max(
            torch.log(1 + F.relu(outputs)) * encoded.attention_mask.unsqueeze(-1), dim=1
        )[0].squeeze()

        cols = vec.nonzero().squeeze().cpu().tolist()
        weights = vec[cols].cpu().tolist()
        terms = [self.idx2token[idx] for idx in cols]
        sparse_dict = dict(zip(terms, weights))

        return sparse_dict


if __name__ == "__main__":
    splade = SPLADE()
    vector = splade(["There once was a man from Nantucket."])

    print(vector)
