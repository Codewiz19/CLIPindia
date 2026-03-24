import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, SiglipModel


class VisionOnly(nn.Module):
    def __init__(self, model: SiglipModel):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_out = self.model.vision_model(pixel_values=pixel_values)

        if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
            x = vision_out.pooler_output
        elif isinstance(vision_out, (tuple, list)) and len(vision_out) > 1:
            x = vision_out[1]
        elif hasattr(vision_out, "last_hidden_state") and vision_out.last_hidden_state is not None:
            x = vision_out.last_hidden_state[:, 0]
        else:
            raise RuntimeError("Unsupported vision output structure for SigLIP")

        if hasattr(self.model, "visual_projection") and self.model.visual_projection is not None:
            x = self.model.visual_projection(x)

        return F.normalize(x, dim=-1)


class TextOnly(nn.Module):
    def __init__(self, model: SiglipModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_out = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
            x = text_out.pooler_output
        elif isinstance(text_out, (tuple, list)) and len(text_out) > 1:
            x = text_out[1]
        elif hasattr(text_out, "last_hidden_state") and text_out.last_hidden_state is not None:
            x = text_out.last_hidden_state[:, 0]
        else:
            raise RuntimeError("Unsupported text output structure for SigLIP")

        if hasattr(self.model, "text_projection") and self.model.text_projection is not None:
            x = self.model.text_projection(x)

        return F.normalize(x, dim=-1)


def export_vision_onnx(model_dir: Path, output_path: Path, image_size: int = 224) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiglipModel.from_pretrained(str(model_dir)).to(device).eval()
    vision_model = VisionOnly(model).to(device).eval()

    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    torch.onnx.export(
        vision_model,
        (dummy,),
        str(output_path),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeds": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"ONNX exported to: {output_path}")


def export_both_onnx(
    model_dir: Path,
    processor_dir: Path,
    vision_output_path: Path,
    text_output_path: Path,
    image_size: int = 224,
    sample_text: str = "a photo of a red kurta",
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiglipModel.from_pretrained(str(model_dir)).to(device).eval()
    processor = AutoProcessor.from_pretrained(str(processor_dir))

    # Vision export
    vision_model = VisionOnly(model).to(device).eval()
    dummy_pixels = torch.randn(1, 3, image_size, image_size, device=device)
    torch.onnx.export(
        vision_model,
        (dummy_pixels,),
        str(vision_output_path),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeds": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Vision ONNX exported to: {vision_output_path}")

    # Text export
    text_model = TextOnly(model).to(device).eval()
    dummy_text = processor(text=[sample_text], return_tensors="pt", padding=True, truncation=True)
    input_ids = dummy_text["input_ids"].to(device)
    attention_mask = dummy_text.get("attention_mask", torch.ones_like(input_ids)).to(device)

    torch.onnx.export(
        text_model,
        (input_ids, attention_mask),
        str(text_output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "text_embeds": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Text ONNX exported to: {text_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SigLIP vision and text encoders to ONNX")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to Hugging Face model folder (e.g., ./hf_model)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("siglip_vision.onnx"),
        help="Vision ONNX output file path",
    )
    parser.add_argument(
        "--processor-dir",
        type=Path,
        default=None,
        help="Path to Hugging Face processor folder (e.g., ./hf_processor). Defaults to --model-dir if not provided.",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=Path("siglip_text.onnx"),
        help="Text ONNX output file path",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size (default: 224)",
    )
    parser.add_argument(
        "--sample-text",
        type=str,
        default="a photo of a red kurta",
        help="Sample text used to create dummy text inputs for ONNX export",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    processor_dir = args.processor_dir if args.processor_dir is not None else args.model_dir
    export_both_onnx(
        model_dir=args.model_dir,
        processor_dir=processor_dir,
        vision_output_path=args.output,
        text_output_path=args.text_output,
        image_size=args.image_size,
        sample_text=args.sample_text,
    )
