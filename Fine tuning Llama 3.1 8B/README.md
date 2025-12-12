# Fine-tuning Llama 3.1 8B with Unsloth

This project demonstrates how to fine-tune the Llama 3.1 8B model using the Unsloth library and the Alpaca dataset.

## Prerequisites

- NVIDIA GPU with at least 8GB VRAM (recommended).
- Python 3.10+
- Windows (with WSL2 recommended) or Linux.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd "Fine tuning Llama 3.1 8B"
    ```

2.  **Install Dependencies:**
    It is highly recommended to use a virtual environment (Conda or venv).
    
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Unsloth installation on Windows can sometimes be tricky. If `pip install` fails, please refer to the [Unsloth installation guide](https://github.com/unslothai/unsloth) for Windows specifics.*

## Usage

### 1. Training

To fine-tune the model, run the training script. This script uses the `yahma/alpaca-cleaned` dataset.

```bash
python src/train.py
```

- This will download the base model (`unsloth/Meta-Llama-3.1-8B-bnb-4bit`).
- It will train for a few steps (configured to 60 steps in the script for demonstration).
- The fine-tuned LoRA adapters will be saved to the `lora_model` directory.

### 2. Inference

To test your fine-tuned model:

```bash
python src/inference.py
```

- Enter an instruction when prompted (e.g., "Tell me a joke about AI").
- The model will generate a response.
