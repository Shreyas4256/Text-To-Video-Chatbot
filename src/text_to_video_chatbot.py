"""Simple Text-to-Video Chatbot.

This script provides a minimal conversational loop that uses a text-generation
model to produce chat replies and a text-to-video diffusion pipeline to turn
prompts into short video clips. A basic Gradio interface is optionally
provided for ease of use.
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import DiffusionPipeline

try:
    import gradio as gr
except ImportError:  # pragma: no cover - gradio is optional
    gr = None

# Default models; these are reasonably small compared to alternatives but still
# provide good results. Users can override them via CLI arguments if desired.
DEFAULT_CHAT_MODEL = "microsoft/DialoGPT-medium"
DEFAULT_VIDEO_MODEL = "damo-vilab/text-to-video-ms-1.7b"

def load_chatbot(model_name: str = DEFAULT_CHAT_MODEL):
    """Load a text generation pipeline for conversation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def load_video_pipeline(model_name: str = DEFAULT_VIDEO_MODEL):
    """Load a text-to-video diffusion pipeline."""
    return DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

def generate_reply(pipeline, chat_history, prompt, max_new_tokens=50):
    """Generate a reply using the chat pipeline."""
    input_text = "\n".join(chat_history + [prompt])
    outputs = pipeline(input_text, max_new_tokens=max_new_tokens)
    text = outputs[0]["generated_text"][len(input_text) :].strip()
    return text

def generate_video(pipeline, prompt, output_dir="videos", frames=16, fps=8):
    """Generate a short video clip from the prompt and save it."""
    os.makedirs(output_dir, exist_ok=True)
    result = pipeline(prompt, num_frames=frames)
    video_path = Path(output_dir) / f"video_{len(os.listdir(output_dir))}.mp4"
    result["images"].save(str(video_path), fps=fps)
    return str(video_path)

def chat_loop(args):
    chat_pipe = load_chatbot(args.chat_model)
    video_pipe = load_video_pipeline(args.video_model)

    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            break
        reply = generate_reply(chat_pipe, chat_history, user_input)
        chat_history.extend([user_input, reply])
        print(f"Bot: {reply}")
        video = generate_video(video_pipe, user_input)
        print(f"Video saved to: {video}")


def launch_gradio(args):
    chat_pipe = load_chatbot(args.chat_model)
    video_pipe = load_video_pipeline(args.video_model)

    history = []

    def respond(message):
        reply = generate_reply(chat_pipe, history, message)
        history.extend([message, reply])
        video = generate_video(video_pipe, message)
        return reply, video

    with gr.Blocks() as demo:
        gr.Markdown("## Text to Video Chatbot")
        chat = gr.ChatInterface(respond, examples=["Hello!"], title="Text2Video Chatbot")
    demo.launch()


def main():
    parser = argparse.ArgumentParser(description="Text to Video Chatbot")
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--video-model", default=DEFAULT_VIDEO_MODEL)
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio interface")
    args = parser.parse_args()

    if args.gradio:
        if gr is None:
            raise SystemExit("gradio is not installed. Please pip install gradio")
        launch_gradio(args)
    else:
        chat_loop(args)


if __name__ == "__main__":  # pragma: no cover
    main()
