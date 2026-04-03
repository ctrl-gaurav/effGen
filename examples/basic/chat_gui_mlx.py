"""
Chat GUI for testing MLX models on Apple Silicon.

A Gradio-based chat interface with streaming, adjustable parameters,
and multiple model support.

Requirements:
    pip install gradio mlx-lm

Usage:
    python examples/basic/chat_gui_mlx.py
    python examples/basic/chat_gui_mlx.py --model LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit
    python examples/basic/chat_gui_mlx.py --port 7861 --share
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterator

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MLX model wrapper
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_model_name = None


def load_mlx_model(model_id: str) -> str:
    """Load an MLX model. Returns status string."""
    global _model, _tokenizer, _model_name

    if _model is not None:
        del _model, _tokenizer
        _model = _tokenizer = None
        import gc; gc.collect()

    try:
        from mlx_lm import load
    except ImportError:
        return "Error: mlx-lm not installed. Run: pip install mlx-lm"

    try:
        _model, _tokenizer = load(model_id)
        _model_name = model_id
        return f"Loaded: {model_id}"
    except Exception as e:
        return f"Error loading model: {e}"


def chat_respond(
    message: str,
    history: list[dict],
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
) -> Iterator[str]:
    """Gradio ChatInterface response function with streaming."""
    if _model is None or _tokenizer is None:
        yield "Model not loaded. Enter a model ID above and click Load Model."
        return

    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    # Build messages for chat template
    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    # Apply chat template
    try:
        formatted = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted = "\n".join(m["content"] for m in messages)

    sampler = make_sampler(
        temp=float(temperature),
        top_p=float(top_p),
    )
    # Note: repetition_penalty is applied via logits_processors in mlx-lm,
    # but make_sampler doesn't support it directly. Using sampler only.

    text_so_far = ""
    for response in stream_generate(
        _model, _tokenizer,
        prompt=formatted,
        max_tokens=int(max_tokens),
        sampler=sampler,
    ):
        text_so_far += response.text
        yield text_so_far


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app(default_model: str):
    import gradio as gr

    with gr.Blocks(title="effGen MLX Chat") as app:
        gr.Markdown("# effGen — MLX Chat\nTest MLX models on Apple Silicon with streaming.")

        # --- Model loading bar ---
        with gr.Row():
            model_id = gr.Textbox(
                value=default_model,
                label="Model ID",
                scale=4,
                placeholder="e.g. LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit",
            )
            load_btn = gr.Button("Load Model", variant="primary", scale=1)
            status = gr.Textbox(label="Status", interactive=False, scale=2)

        load_btn.click(fn=load_mlx_model, inputs=[model_id], outputs=[status])

        # --- Parameters (above chat so they're accessible as additional_inputs) ---
        with gr.Accordion("Generation Parameters", open=False):
            system_prompt = gr.Textbox(
                value="You are a helpful assistant.",
                label="System Prompt",
                lines=2,
            )
            with gr.Row():
                temperature = gr.Slider(
                    0.0, 2.0, value=0.1, step=0.05, label="Temperature"
                )
                top_p = gr.Slider(
                    0.0, 1.0, value=0.1, step=0.05, label="Top-p"
                )
            with gr.Row():
                max_tokens = gr.Slider(
                    64, 4096, value=512, step=64, label="Max Tokens"
                )
                rep_penalty = gr.Slider(
                    1.0, 2.0, value=1.05, step=0.01, label="Repetition Penalty"
                )

        # --- Chat interface ---
        gr.ChatInterface(
            fn=chat_respond,
            additional_inputs=[
                system_prompt, temperature, top_p, max_tokens, rep_penalty,
            ],
            chatbot=gr.Chatbot(height=480),
        )

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MLX Chat GUI")
    parser.add_argument(
        "--model",
        default="LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit",
        help="Default model ID to pre-fill",
    )
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--autoload", action="store_true", help="Load model on startup")
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError:
        print("Gradio is required. Install with:")
        print("  pip install gradio")
        sys.exit(1)

    if args.autoload:
        print(f"Loading model: {args.model}")
        result = load_mlx_model(args.model)
        print(result)

    app = build_app(args.model)
    app.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css="footer { display: none !important; }",
    )


if __name__ == "__main__":
    main()
