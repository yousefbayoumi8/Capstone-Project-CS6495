import sys
from src.model_loader import load_model
from src.target_model import run_chatbot

if __name__ == "__main__":
    
    # Choose model from command line: python main.py qwen
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    
    model, tokenizer = load_model(model_name)
    run_chatbot(model, tokenizer)