"""
Gradio chat UI for the SecureBank prompt-injection target.

Run on desktop, access from laptop over Tailscale.
Reuses src/model_loader.py and the SYSTEM_PROMPT from src/target_model.py
so the chatbot under attack matches the CLI version exactly.

Usage:
    python webui.py                          # default model (qwen), no auth
    python webui.py --model llama
    python webui.py --model qwen --auth user:pass
    python webui.py --port 7860 --host 0.0.0.0
"""

import argparse
import os

import gradio as gr

from src.model_loader import load_model, generate_response
from src.target_model import SYSTEM_PROMPT


def build_app(model, tokenizer, model_name: str):

    def _to_text(content):
        # Gradio sometimes stores message content as multimodal parts:
        # [{"type": "text", "text": "..."}] — flatten to plain string.
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    parts.append(p.get("text") or p.get("content") or "")
                elif isinstance(p, str):
                    parts.append(p)
            return "".join(parts)
        return str(content)

    def respond(message, history: list):
        # Normalize Gradio history to OpenAI-style dicts with string content.
        normalized = []
        for item in history:
            if isinstance(item, dict):
                normalized.append({
                    "role": item["role"],
                    "content": _to_text(item["content"]),
                })
            else:
                user_msg, bot_msg = item
                if user_msg:
                    normalized.append({"role": "user", "content": _to_text(user_msg)})
                if bot_msg:
                    normalized.append({"role": "assistant", "content": _to_text(bot_msg)})
        messages = normalized + [{"role": "user", "content": _to_text(message)}]
        return generate_response(model, tokenizer, SYSTEM_PROMPT, messages)

    with gr.Blocks(title=f"SecureBank Assistant ({model_name})") as app:
        gr.Markdown(f"# SecureBank Assistant\n*Backend: `{model_name}` — prompt-injection target*")

        chat = gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(height=550),
            textbox=gr.Textbox(placeholder="Ask SecureBank a question...", scale=7),
            submit_btn=True,
        )

        with gr.Accordion("Demonstrator controls (do not share with attackers)", open=False):
            gr.Markdown("**Active system prompt:**")
            gr.Code(value=SYSTEM_PROMPT, language=None, interactive=False)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen", choices=["qwen", "llama", "gemma"],
                        help="Which model to load (default: qwen)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address. 0.0.0.0 = reachable on Tailscale. 127.0.0.1 = local only.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--auth", default=None,
                        help="Optional 'user:password' for basic auth. Tailscale is already private; "
                             "use this only if you want defense-in-depth.")
    parser.add_argument("--share", action="store_true",
                        help="Create a public gradio.live URL. Not needed if you have Tailscale.")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    app = build_app(model, tokenizer, args.model)

    auth = None
    if args.auth:
        user, _, pw = args.auth.partition(":")
        if not user or not pw:
            raise SystemExit("--auth must be in the form 'user:password'")
        auth = (user, pw)

    app.queue().launch(
        server_name=args.host,
        server_port=args.port,
        auth=auth,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()