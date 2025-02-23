# mtcs-pytorch
A flexible Monte Carlo Tree Search framework with PyTorch for decision-making in language models.

![MCTS Van Gogh-like painting](image_fx_-2.jpg)

## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage

To use the MCTS framework, follow the example provided in the `example/mcts.ipynb` notebook.

## Example

The following example demonstrates how to load a model and tokenizer, create an input prompt, configure MCTS, and run the MCTS:

```python
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.mcts import MCTS

sys.path.append('..')

# Load model and tokenizer
path = 'mkurman/llama-3.2-MEDIT-3B-o1'
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(path)

# Create input prompt
chat_like_texts = [
    [
        {
            "role": "user",
            "content": "Question: A 48-year-old man comes to the clinic because of a 10-year history of recurrent, intrusive thoughts that his house will be broken into and damaged by criminals or accidentally destroyed by a fire when he is not home. These thoughts have worsened during the past 2 months. He reports now spending 4 hours daily checking that the doors and windows are closed and locked and that the stove and oven are turned off; he previously spent 2 hours daily doing these tasks. He says he cannot keep a job or leave the house very much because of the amount of time he spends checking these things. He has no other history of serious illness and takes no medications. Physical examination shows no abnormalities. On mental status examination, he has an anxious mood and a sad affect. He is fully oriented. He is not having hallucinations or delusions. The most effective pharmacotherapy for this patient is an agent that targets which of the following neurotransmitters?\nA. γ-Aminobutyric acid\nB. Dopamine\nC. Glutamate\nD. Norepinephrine\nE. Serotonin",
        },
    ]
]

input_ids = tokenizer(tokenizer.apply_chat_template(chat_like_texts[0], tokenize=False, add_generation_prompt=True), padding=True, truncation=True, return_tensors="pt")

# Configure MCTS
MAX_DEPTH = 6
MAX_SIMULATIONS = 4
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.5
MAX_TOTAL_TOKENS = 8192

# Run the MCTS
model.to("cuda")
input_tokens = input_ids['input_ids'].to('cuda')
model.eval()

print(tokenizer.decode(input_tokens[0]))

while len(input_tokens[0]) < MAX_TOTAL_TOKENS:
    mcts = MCTS(model, tokenizer, max_depth=MAX_DEPTH, num_simulations=MAX_SIMULATIONS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS, stop_tokens=model.config.eos_token_id)
    
    new_tokens = mcts.search(input_tokens)

    new_tokens = new_tokens[..., input_tokens.shape[-1]:]

    input_tokens = torch.cat([input_tokens, new_tokens], dim=-1)

    print(tokenizer.decode(new_tokens[0]), end='')

    if has_eos(new_tokens, eos_token=model.config.eos_token_id):
        break
```
