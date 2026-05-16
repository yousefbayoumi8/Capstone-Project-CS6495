import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model IDs on HuggingFace
MODELS = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it"
}

def load_model(model_name: str):
    """
    Load a model and its tokenizer.
    model_name: one of 'llama', 'qwen', 'gemma'
    """
    
    if model_name not in MODELS:
        raise ValueError(f"Unknown model. Choose from: {list(MODELS.keys())}")
    
    model_id = MODELS[model_name]
    print(f"Loading {model_name} from {model_id}...")
    print("This will download the model on first run (several GB).")
    
    # Tokenizer is the same for all three
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Gemma needs special handling — slightly over 16GB at float16
    # so we let it overflow to CPU RAM automatically
    if model_name == "gemma":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"  # fills GPU first, overflows to CPU RAM
        )
    
    # Llama and Qwen fit in 16GB at float16
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    model.eval()  # set to evaluation mode — disables dropout etc
    print(f"{model_name} loaded successfully.")
    
    return model, tokenizer


def generate_response(model, tokenizer, system_prompt: str, 
                      messages: list, max_new_tokens: int = 512):
    """
    Generate a response given a system prompt and conversation history.
    
    messages: list of dicts like [{"role": "user", "content": "hello"}]
    Returns: string response
    """
    
    # Build full conversation
    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages += messages
    
    # Convert to token IDs using the model's chat template
    # Each model has its own special format — apply_chat_template handles this
    inputs = tokenizer.apply_chat_template(
        full_messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    new_tokens = output_ids[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()