from unsloth import FastLanguageModel
import torch

# 1. Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True

# 2. Load Fine-tuned Model
model_path = "lora_model" # Path to the saved LoRA adapter
print(f"Loading model from: {model_path}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# 3. Predict
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

def generate_response(instruction, input_text=""):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction, # instruction
                input_text, # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    decoded_output = tokenizer.batch_decode(outputs)
    return decoded_output[0]

if __name__ == "__main__":
    print("Inference ready. Type 'exit' to quit.")
    while True:
        instruction = input("Enter Instruction: ")
        if instruction.lower() == "exit":
            break
        input_text = input("Enter Input (optional): ")
        
        print("\nGenerating response...")
        response = generate_response(instruction, input_text)
        print("\n=== RESPONSE ===")
        # Basic parsing to show only the response part could be added, but printing full outline for now
        print(response)
        print("================\n")
