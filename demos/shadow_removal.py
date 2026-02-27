"""Shadow Removal Gradio demo.

Standalone usage:
    python demos/shadow_removal.py [--model-path pretrained_models/] [--share] [--port 7860]

Also importable as a tab component via build_shadow_ui().
"""

import argparse
import os

import gradio as gr
import torch

from crocodilight.inference import (
    get_device,
    load_model,
    load_mapper,
    pil_to_tensor,
    tensor_to_pil, unpad, pad_to_min_size,
)


HF_REPO_ID = "alistairfoggin/CroCoDiLight"

WEIGHT_FILES = {
    "model": "CroCoDiLight.pth",
    "shadow_mapper": "CroCoDiLight_shadow_mapper.pth",
    "albedo_mapper": "CroCoDiLight_albedo_mapper.pth",
}


def get_weight_path(key, local_dir="pretrained_models"):
    """Check local path first, fall back to HF Hub download."""
    filename = WEIGHT_FILES[key]
    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        return local_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)


def load_shadow_models(model_path="pretrained_models", device=None):
    """Load the base model and shadow mapper."""
    if device is None:
        device = get_device()
    model = load_model(get_weight_path("model", model_path), device)
    mapper = load_mapper(model, get_weight_path("shadow_mapper", model_path), device)
    return model, mapper, device


def build_shadow_ui(model, mapper, device):
    """Build the Shadow Removal tab UI. Returns a gr.Blocks component."""

    def run_shadow_inference(image, resize):
        if image is None:
            raise gr.Error("Please upload an image.")
        resize = int(resize) if resize is not None and resize > 0 else None
        try:
            img_tensor = pil_to_tensor(image, device, resize=resize)
            img_tensor, pad_info = pad_to_min_size(img_tensor)
            with torch.no_grad():
                result = model.apply_mapper(img_tensor, mapper, use_consistency=False)
            result = unpad(result, pad_info)
            return tensor_to_pil(result)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise gr.Error("GPU ran out of memory. Please try a smaller image.")

    with gr.Blocks() as shadow_ui:
        gr.Markdown("## Shadow Removal\nUpload an image to remove shadows.")
        with gr.Row():
            input_image = gr.Image(type="pil", label="Input Image")
            output_image = gr.Image(type="pil", label="Result", format="png", interactive=False)
        resize_input = gr.Number(value=None, label="Resize (leave empty for original size)", precision=0)
        run_btn = gr.Button("Remove Shadows", variant="primary")
        run_btn.click(fn=run_shadow_inference, inputs=[input_image, resize_input], outputs=output_image)

    return shadow_ui


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shadow Removal Gradio Demo")
    parser.add_argument(
        "--model-path",
        default="pretrained_models",
        help="Directory containing weight files",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for inference (cuda:0, cpu). Auto-detects if not set.",
    )
    parser.add_argument(
        "--share", action="store_true", help="Launch with public Gradio share link"
    )
    parser.add_argument("--port", type=int, default=7860, help="Local server port")
    args = parser.parse_args()

    device = get_device(args.device)
    model, mapper, device = load_shadow_models(args.model_path, device)
    demo = build_shadow_ui(model, mapper, device)
    demo.launch(share=args.share, server_port=args.port)
