# LoRA Testing Toolkit

A streamlined Python toolkit for generating customized images using Flux models integrated with LoRA adapters directly from Hugging Face. Ideal for rapid experimentation and evaluation of fine-tuned image generation.

## Features

- **Automated LoRA Integration:** Seamlessly download and load `.safetensors` LoRA models from Hugging Face.
- **Flux Pipelines:** Utilize powerful Flux models for high-quality text-to-image and image-to-image generation.
- **Multi-LoRA Support:** Integrate and manage multiple LoRA adapters effortlessly.
- **Remote Storage:** Directly upload generated images to cloud storage (e.g., Google Drive via `rclone`).

## Installation

Clone the repository:

```bash
git clone https://github.com/dhawan98/lora-testing-toolkit.git
cd lora-testing-toolkit

Install required dependencies:     

pip install torch diffusers transformers peft huggingface_hub pillow

