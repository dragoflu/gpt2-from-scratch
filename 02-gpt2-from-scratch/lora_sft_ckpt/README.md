---
library_name: peft
tags:
- lora
---

# LoRA adapter for the hand-written GPT-2 (124M)

Trained with `reproducing_gpt-2.py` on `tatsu-lab/alpaca`, prompt tokens masked out of the loss.
Config: `r=8`, `lora_alpha=16`, `lora_dropout=0.05`, targets `c_attn` / `c_proj` / `c_fc`, lr `1e-4`.

The base model is not a Hugging Face `GPT2LMHeadModel` but the local `GPT2` class, which uses the
same parameter names. Load it as:

```python
base = load_pretrained('gpt2')
gpt = PeftModel.from_pretrained(base, CKPT_DIR)
```

`history.json` holds the train/val loss curves plotted in the root README.
