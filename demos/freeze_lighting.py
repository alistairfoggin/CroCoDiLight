"""Lighting Freeze Gradio demo.

Applies the lighting from a reference image to content image(s).
Supports single-image mode (HF Space + local) and batch mode (local only).

Standalone usage:
    python demos/freeze_lighting.py [--model-path pretrained_models/] [--share] [--port 7860]

Also importable as a tab component via build_freeze_ui().
"""

import argparse
import io
import os
import zipfile

import gradio as gr
import torch
from PIL import Image

from crocodilight.inference import (
    get_device,
    load_model,
    extract_features_pil,
    tensor_to_pil,
)

# Conditional HF Spaces GPU decorator
try:
    import spaces

    gpu_decorator = spaces.GPU
    IS_HF_SPACE = True
except (ImportError, ModuleNotFoundError):

    def gpu_decorator(fn):
        return fn

    IS_HF_SPACE = False


HF_REPO_ID = "alistairfoggin/CroCoDiLight"

WEIGHT_FILES = {
    "model": "CroCoDiLight.pth",
}


def get_weight_path(key, local_dir="pretrained_models"):
    """Check local path first, fall back to HF Hub download."""
    filename = WEIGHT_FILES[key]
    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        return local_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)


def load_freeze_model(model_path="pretrained_models", device=None):
    """Load the base model (no mapper needed for lighting freeze)."""
    if device is None:
        device = get_device()
    model = load_model(get_weight_path("model", model_path), device)
    return model, device


def _relight_single(model, device, ref_image, content_image):
    """Apply lighting from ref_image to content_image. Returns PIL Image."""
    _, dyn_ref, _, _ = extract_features_pil(model, ref_image, device)
    static, _, pos, tiling_module = extract_features_pil(model, content_image, device)
    with torch.no_grad():
        feat = model.lighting_entangler(static, pos, dyn_ref)
        img_info = {"height": 448, "width": 448}
        out_img = model.croco.decode(feat, pos, img_info)
        out_img = tiling_module.rebuild_with_masks(out_img)
    return tensor_to_pil(out_img)


def build_freeze_ui(model, device):
    """Build the Lighting Freeze tab UI. Returns a gr.Blocks component."""

    @gpu_decorator
    def run_freeze_single(ref_image, content_image):
        if ref_image is None:
            raise gr.Error("Please upload a lighting reference image.")
        if content_image is None:
            raise gr.Error("Please upload a content image.")
        return _relight_single(model, device, ref_image, content_image)

    def run_freeze_batch(ref_image, content_images):
        if ref_image is None:
            raise gr.Error("Please upload a lighting reference image.")
        if not content_images:
            raise gr.Error("Please upload at least one content image.")

        results = []
        for img_path in content_images:
            content_img = Image.open(img_path).convert("RGB")
            result = _relight_single(model, device, ref_image, content_img)
            results.append(result)

        # Create downloadable zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, img in enumerate(results):
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                zf.writestr(f"relit_{i:04d}.png", img_buffer.getvalue())
        zip_buffer.seek(0)
        zip_path = os.path.join(os.path.dirname(__file__), "relit_batch.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_buffer.getvalue())

        return results, zip_path

    with gr.Blocks() as freeze_ui:
        gr.Markdown(
            "## Lighting Freeze\n"
            "Apply the lighting from a reference image to content image(s).\n"
            "The reference provides lighting; the content image keeps its structure."
        )

        if not IS_HF_SPACE:
            mode = gr.Radio(
                choices=["Single Image", "Batch"],
                value="Single Image",
                label="Mode",
            )

        # --- Single image mode ---
        with gr.Group(visible=True) as single_group:
            with gr.Row():
                ref_image = gr.Image(type="pil", label="Lighting Reference")
                content_image = gr.Image(type="pil", label="Content Image")
            single_output = gr.Image(type="pil", label="Result", interactive=False)
            single_btn = gr.Button("Apply Lighting", variant="primary")
            single_btn.click(
                fn=run_freeze_single,
                inputs=[ref_image, content_image],
                outputs=single_output,
            )

        # --- Batch mode (local only) ---
        if not IS_HF_SPACE:
            with gr.Group(visible=False) as batch_group:
                with gr.Row():
                    ref_image_batch = gr.Image(type="pil", label="Lighting Reference")
                    content_images_batch = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Content Images",
                    )
                batch_gallery = gr.Gallery(label="Results")
                batch_download = gr.File(label="Download All (ZIP)", interactive=False)
                batch_btn = gr.Button("Apply Lighting (Batch)", variant="primary")
                batch_btn.click(
                    fn=run_freeze_batch,
                    inputs=[ref_image_batch, content_images_batch],
                    outputs=[batch_gallery, batch_download],
                )

            def toggle_mode(mode_val):
                return (
                    gr.update(visible=mode_val == "Single Image"),
                    gr.update(visible=mode_val == "Batch"),
                )

            mode.change(
                fn=toggle_mode,
                inputs=mode,
                outputs=[single_group, batch_group],
            )

    return freeze_ui


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lighting Freeze Gradio Demo")
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
    model, device = load_freeze_model(args.model_path, device)
    demo = build_freeze_ui(model, device)
    demo.launch(share=args.share, server_port=args.port)
