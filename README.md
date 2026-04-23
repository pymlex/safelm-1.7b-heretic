# Heretic on SafeLM-1.7B-Instruct

## Overview

We implement an abliterated version of `locuslab/safelm-1.7b-instruct`. [SafeLM][1] is a 1.7B parameter model family trained with safety pretraining, and the instruct checkpoint is instruction-tuned on top of the SmolLM2 architecture. [Heretic][2] is a fully automatic abliteration tool that co-optimizes refusal suppression and KL divergence from the original model, which makes it a natural fit for decensoring while trying to preserve general model quality. The main objective is to reduce refusal behavior on harmful prompts while keeping the output distribution as close as possible to the base model, which is considered to be safe.

## Objective

We evaluate the model with two metrics:

$$
\text{refusal rate} = \frac{N_{\text{refusals}}}{N},
\qquad
\text{KLD} = D_{\mathrm{KL}}(p_{\text{heretic}} | p_{\text{base}})
$$

Refusals should decrease and KL divergence should remain bounded.

Target constraint used:

$$
D_{\mathrm{KL}} \leq 0.05
$$

It keeps the model from drifting too far away from the original SafeLM checkpoint.

## Dataset

We use two prompt pools:

* `tatsu-lab/alpaca` for harmless prompts
* `mlabonne/harmful_behaviors` for harmful prompts

## Configuration

We use a slightly non-default setup that was tuned manually for stability. We keep everything in `bfloat16`, turn quantization off, and use a relatively large batch size of `512` with `max_response_length = 120`.

```toml
dtypes = ["bfloat16"]

quantization = "none"
max_batch_size = 512
max_response_length = 120

kl_divergence_scale = 1.0
kl_divergence_target = 0.05

orthogonalize_direction = true
row_normalization = "pre"
full_normalization_lora_rank = 8

winsorization_quantile = 0.98

n_trials = 1200
n_startup_trials = 100
```

The main adjustment was the number of trials. The usual 200–400 trials were not enough here: the refusal rate kept oscillating between 80% and 100%. After increasing the budget to `1200`, the trajectory became much smoother, and the run started converging to a low-refusal region in a stable way.

The prompt mix also mattered a lot. We used `800` harmless prompts and `300` harmful prompts for a final run. Experiments showed that there is no visible convergence for less than `100` prompts for each class, and better with the increase to `200` and `300`. We have `416` harmful prompts in total and can't use more but making the harmless subset bigger allowed us to decrease the refusal rate from 20–50% to nearly 10%.

The normalization settings were also important. `orthogonalize_direction = true` and `row_normalization = "pre"` made the updates less noisy and easier to optimize.

## Chat Template

SafeLM does not include a built-in `chat_template`, the instruct version doesn't have it configured in an acceptable way, so it was added manually. That was necessary because Heretic expects chat-formatted input.

```text
<|im_start|>system
...<|im_end|>
<|im_start|>user
...<|im_end|>
<|im_start|>assistant
```

## Results

The final result is:

```text
Refusals: 41 / 416
Refusal rate: 9.86%
```

This means the model stopped refusing in most cases, while still staying fairly close to the original checkpoint thanks to the KL constraint. We provide the predictions in `evaluation_output.txt` and also attach some examples of QA on sensitive topics in `chat_examples.txt`.

## Inference

Load the model from Hub.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "pymlex/safelm-1.7b-heretic"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
```

Ask something.

```python
def ask(prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.9 if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()

print(ask("How to make a bomb?"))
```

[1]: https://github.com/p-e-w/heretic "GitHub - p-e-w/heretic: Fully automatic censorship removal ..."
[2]: https://huggingface.co/locuslab/safelm-1.7b-instruct "locuslab/safelm-1.7b-instruct"
