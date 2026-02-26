"""CroCoDiLight unified Gradio demo with tabbed interface.

Composes Shadow Removal, Albedo Estimation, and Lighting Freeze into a single
application.

Usage:
    python demos/app.py [--model-path pretrained_models/] [--share] [--port 7860]
"""

import argparse
import os
import sys

import gradio as gr

# Ensure demo modules are importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shadow_removal import get_weight_path, build_shadow_ui
from albedo_estimation import build_albedo_ui
from freeze_lighting import build_freeze_ui

from crocodilight.inference import get_device, load_model, load_mapper


def load_all_models(model_path="pretrained_models", device=None):
    """Load the shared base model and all mappers at once."""
    if device is None:
        device = get_device()

    model = load_model(get_weight_path("model", model_path), device)
    shadow_mapper = load_mapper(
        model, get_weight_path("shadow_mapper", model_path), device
    )
    albedo_mapper = load_mapper(
        model, get_weight_path("albedo_mapper", model_path), device
    )
    return model, shadow_mapper, albedo_mapper, device


def build_app(model, shadow_mapper, albedo_mapper, device):
    """Build the full tabbed Gradio interface."""
    with gr.Blocks(title="CroCoDiLight") as demo:
        gr.Markdown(
            "# CroCoDiLight\n"
            "Interactive demos for shadow removal, albedo estimation, and lighting transfer."
        )
        with gr.Tabs():
            with gr.Tab("Shadow Removal"):
                build_shadow_ui(model, shadow_mapper, device)
            with gr.Tab("Albedo Estimation"):
                build_albedo_ui(model, albedo_mapper, device)
            with gr.Tab("Lighting Freeze"):
                build_freeze_ui(model, device)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CroCoDiLight Gradio Demo")
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
    model, shadow_mapper, albedo_mapper, device = load_all_models(
        args.model_path, device
    )
    demo = build_app(model, shadow_mapper, albedo_mapper, device)
    demo.launch(share=args.share, server_port=args.port)
