from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import torch

# Step 1: Define paths
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Full precision model from Hugging Face
adapter_path = "./adapters"  # Path to your LoRA adapter
output_path = "./provilac-llama3-8b"  # Where to save the merged model

# Step 2: Load base model in half precision
print("🔄 Loading base model...")
model = LlamaForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Step 3: Load and apply LoRA adapter
print("🔌 Applying LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)

# Step 4: Merge LoRA weights into base model
print("🧠 Merging LoRA into base model...")
model = model.merge_and_unload()

# Step 5: Save merged model
print(f"💾 Saving merged model to: {output_path}")
model.save_pretrained(output_path)

# Step 6: Save tokenizer (required for loading and generation later)
print("💾 Saving tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(output_path)

print("✅ Done! Your model is ready to use.")
