"""
Type a prompt -> generate an image in the cloud via Hugging Face Inference Providers -> open it in Windows Photos.
Runs on any PC; no GPU needed.

Setup (once):
    py -m pip install -U huggingface_hub pillow
    Create a free token at https://huggingface.co/settings/tokens
    ("Make calls to Inference Providers" permission), then run:
        setx HF_TOKEN hf_yourtokenhere
    and open a new terminal.

Run:
    py image_prompt.py
"""

import os
from datetime import datetime
from pathlib import Path

from huggingface_hub import InferenceClient

MODELS = {
    "1": "black-forest-labs/FLUX.1-schnell",  # fast, good all-rounder
    "2": "Tongyi-MAI/Z-Image-Turbo",           # fast, strong photorealism
    "3": "black-forest-labs/FLUX.1-dev",      # slower, higher quality
    "4": "stabilityai/stable-diffusion-xl-base-1.0",
}

OUTPUT_DIR = Path(__file__).parent / "outputs"


def pick_model():
    print("Models:")
    for key, name in MODELS.items():
        print(f"  {key}. {name}")
    choice = input("Choose a model [1]: ").strip() or "1"
    return MODELS.get(choice, MODELS["1"])


def main():
    token = os.environ.get("HF_TOKEN") or input("Hugging Face token (hf_...): ").strip()
    client = InferenceClient(provider="auto", api_key=token)
    model = pick_model()

    while True:
        prompt = input("\nPrompt (blank to quit): ").strip()
        if not prompt:
            break
        print("Generating...")
        try:
            image = client.text_to_image(prompt, model=model)
        except Exception as e:
            print(f"Failed: {e}")
            continue

        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.png"
        image.save(path)
        print(f"Saved {path}")
        os.startfile(path)  # opens in the default image viewer (Windows Photos)


if __name__ == "__main__":
    main()
