import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils.logger import get_logger

log = get_logger("perplexity")

_device    = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = None
_model     = None

def _load():
    global _tokenizer, _model
    if _model is None:
        log.info(f"Loading GPT-2 on {_device}...")
        _tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        _model     = GPT2LMHeadModel.from_pretrained("gpt2").to(_device)

def get_perplexity(code: str) -> float:
    if not code or len(code) < 20:
        return 0.0
    try:
        _load()
        inputs = _tokenizer(
            code, return_tensors="pt", truncation=True, max_length=512
        ).to(_device)
        with torch.no_grad():
            loss = _model(**inputs, labels=inputs["input_ids"]).loss
        return float(torch.exp(loss).item())
    except Exception as e:
        log.warning(f"Perplexity failed: {e}")
        return 0.0