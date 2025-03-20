import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from transformers import CLIPImageProcessor
from lora_loading_patch import load_lora_into_transformer
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker
)
import os
import random
import string
from huggingface_hub import hf_hub_download, list_repo_files, login
from typing import List
from PIL import Image
from peft import PeftModel

# Authenticate with Hugging Face
login(token="add your own hf token here")  # Replace with your Hugging Face token

# Utility function to generate a random filename
def generate_random_filename(extension="png"):
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"output_{random_str}.{extension}"

# Load model
def load_flux_model():
    model_id = "black-forest-labs/FLUX.1-dev"
    txt2img_pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    img2img_pipe = FluxImg2ImgPipeline(
        transformer=txt2img_pipe.transformer,
        scheduler=txt2img_pipe.scheduler,
        vae=txt2img_pipe.vae,
        text_encoder=txt2img_pipe.text_encoder,
        text_encoder_2=txt2img_pipe.text_encoder_2,
        tokenizer=txt2img_pipe.tokenizer,
        tokenizer_2=txt2img_pipe.tokenizer_2,
    ).to("cuda")

    txt2img_pipe.__class__.load_lora_into_transformer = classmethod(load_lora_into_transformer)
    img2img_pipe.__class__.load_lora_into_transformer = classmethod(load_lora_into_transformer)

    return txt2img_pipe, img2img_pipe

# Automatically find and download the .safetensors file
def download_lora_from_huggingface(repo_id: str):
    files = list_repo_files(repo_id)
    safetensors_file = next((file for file in files if file.endswith('.safetensors')), None)

    if not safetensors_file:
        raise FileNotFoundError(f"No .safetensors file found in {repo_id}")

    # Change download path to current working directory
    custom_cache_dir = os.path.join(os.getcwd(), "huggingface_cache")
    os.makedirs(custom_cache_dir, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=repo_id, 
        filename=safetensors_file, 
        cache_dir=custom_cache_dir  # Custom cache path
    )
    
    print(f"✅ LoRA weights downloaded to: {local_path}")
    return local_path

# Image generation function
def generate_images(
    prompt: str,
    hf_lora: str,
    output_folder: str = "output",
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    num_images: int = 1
) -> List[str]:

    txt2img_pipe, _ = load_flux_model()

    # Load LoRA Weights
    if hf_lora:
        print(f"Loading LoRA weights from: {hf_lora}")
        lora_path = download_lora_from_huggingface(hf_lora)
        txt2img_pipe.load_lora_weights(lora_path, adapter_name="default_adapter", low_cpu_mem_usage=True)
        print(f"✅ LoRA weights successfully loaded with PEFT from {lora_path}")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate images
    output_paths = []
    for i in range(num_images):
        image = txt2img_pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]

        # Save image locally
        local_path = os.path.join(output_folder, generate_random_filename())
        image.save(local_path)

        output_paths.append(local_path)
        print(f"✅ Image saved locally to: {local_path}")

    return output_paths

# Example Usage
if __name__ == "__main__":
    prompt = "Aashish as superman flying above New York"

    # Run inference
    generate_images(
        prompt=prompt,
        hf_lora="aashudhawan/lora_face_aashish",  # LoRA model directly from Hugging Face
        output_folder="output",  # Local 'output' folder
        width=512,
        height=512,
        guidance_scale=7.5,
        num_inference_steps=50,
        num_images=3
    )
