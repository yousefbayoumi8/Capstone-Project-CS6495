import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Model IDs on HuggingFace
MODELS = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it"
}

# NF4 4-bit quantization: ~75% VRAM reduction, ~2% quality drop.
# Lets a 16 GB GPU comfortably hold any of the three 7-9B models plus KV cache.
_BNB_4BIT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def load_model(model_name: str, quantize: bool = True):
    """
    Load a model and its tokenizer.
    model_name: one of 'llama', 'qwen', 'gemma'
    quantize:   when True (default), load weights in 4-bit via bitsandbytes.
                Set False to load in FP16 (needs ~16 GB VRAM for 7-9B models).
    """

    if model_name not in MODELS:
        raise ValueError(f"Unknown model. Choose from: {list(MODELS.keys())}")

    model_id = MODELS[model_name]
    precision = "4-bit (bitsandbytes)" if quantize else "FP16"
    print(f"Loading {model_name} from {model_id} ({precision})...")
    print("This will download the model on first run (several GB).")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    load_kwargs = {"device_map": "auto"}
    if quantize:
        load_kwargs["quantization_config"] = _BNB_4BIT
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    model.eval()  # disables dropout etc.
    print(f"{model_name} loaded successfully.")

    return model, tokenizer


def render_chat(tokenizer, messages: list):
    """
    Apply the tokenizer's chat template. If the model's template doesn't
    support a `system` role (e.g. Gemma), retry with the system content
    merged into the first user message.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except Exception as e:
        if "system" not in str(e).lower():
            raise
        sys_text = "\n\n".join(
            m["content"] for m in messages if m.get("role") == "system"
        )
        rest = [m for m in messages if m.get("role") != "system"]
        if sys_text and rest and rest[0].get("role") == "user":
            rest = [{"role": "user",
                     "content": f"{sys_text}\n\n{rest[0]['content']}"}] + rest[1:]
        elif sys_text:
            rest = [{"role": "user", "content": sys_text}] + rest
        return tokenizer.apply_chat_template(
            rest,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

def generate_response(model, tokenizer, system_prompt: str,
                      messages: list, max_new_tokens: int = 512,
                      temperature: float = 0.7):
    """
    Generate a response given a system prompt and conversation history.

    messages: list of dicts like [{"role": "user", "content": "hello"}]
    Returns: string response
    """

    # Build full conversation
    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages += messages

    # Convert to token IDs using the model's chat template (with Gemma fallback)
    inputs = render_chat(tokenizer, full_messages).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    new_tokens = output_ids[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()