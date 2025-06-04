# Text-To-Video-Chatbot

This project provides a simple chatbot that replies using a text-generation
model and creates short video clips from the conversation prompts using a
text-to-video diffusion pipeline. A small Gradio interface is included for easy
interaction.

## Setup

Install the required dependencies:

```bash
pip install transformers diffusers accelerate gradio torch
```

> The diffusion pipeline downloads large model weights on first run. Ensure you
> have enough disk space and a GPU for best performance.

## Usage

### Command line

Run the chatbot in your terminal:

```bash
python src/text_to_video_chatbot.py
```

Type messages and the bot will respond while saving generated videos in the
`videos/` directory. Enter `quit` or `exit` to stop.

### Gradio interface

Launch a simple web UI:

```bash
python src/text_to_video_chatbot.py --gradio
```

Open the displayed link in your browser and chat with the bot; generated videos
are shown alongside the responses.
