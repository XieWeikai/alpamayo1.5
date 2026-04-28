"""Run Alpamayo 1.5 inference with synthetic (dummy) data.

This script loads the model from a local checkpoint and runs inference
using randomly generated images and trajectory data, bypassing the need
for the PhysicalAI-AV HuggingFace dataset.

NOTE: nvidia/Cosmos-Reason2-8B is a gated repo. We override vlm_name_or_path
to use Qwen/Qwen3-VL-8B-Instruct (same architecture, publicly accessible)
for the processor and VLM config initialization.
"""

import json
import sys
import torch
import numpy as np

sys.path.insert(0, "src")

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.config import Alpamayo1_5Config

MODEL_PATH = "/data-25T/models/Alpamayo-1.5-10B"
VLM_PROCESSOR = "Qwen/Qwen3-VL-8B-Instruct"
DEVICE = "cuda"


def make_dummy_data(
    num_cameras: int = 4,
    num_frames: int = 4,
    img_h: int = 480,
    img_w: int = 640,
    num_history_steps: int = 16,
) -> dict:
    """Create synthetic input data mimicking the real dataset format."""
    image_frames = torch.randint(
        0, 256, (num_cameras, num_frames, 3, img_h, img_w), dtype=torch.uint8
    )
    camera_indices = torch.tensor([0, 1, 2, 6], dtype=torch.int64)

    ego_history_xyz = torch.zeros(1, 1, num_history_steps, 3, dtype=torch.float32)
    for t in range(num_history_steps):
        ego_history_xyz[0, 0, t, 0] = t * 0.5

    ego_history_rot = (
        torch.eye(3, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, 1, num_history_steps, 3, 3)
        .clone()
    )

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }


def load_model():
    """Load the Alpamayo1_5 model, working around the gated Cosmos-Reason2 repo."""
    # Load config and override vlm_name_or_path to avoid gated repo
    with open(f"{MODEL_PATH}/config.json") as f:
        config_dict = json.load(f)
    config_dict["vlm_name_or_path"] = VLM_PROCESSOR
    config_dict["attn_implementation"] = "sdpa"
    config = Alpamayo1_5Config(**config_dict)

    model = Alpamayo1_5.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    return model.to(DEVICE)


def run_trajectory_inference(model, processor, data):
    """Run the full trajectory prediction pipeline."""
    print("\n=== Trajectory Prediction Mode ===")
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"],
    )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, DEVICE)

    torch.cuda.manual_seed_all(42)
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    print(f"pred_xyz shape: {pred_xyz.shape}")
    print(f"pred_rot shape: {pred_rot.shape}")
    print(f"Chain-of-Causation:\n{extra['cot'][0]}")


def run_vqa_inference(model, processor, data):
    """Run VQA (text generation) mode."""
    print("\n=== VQA Mode ===")
    messages = helper.create_vqa_message(
        frames=data["image_frames"].flatten(0, 1),
        question="Describe the driving scene.",
        camera_indices=data["camera_indices"],
    )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {"tokenized_data": inputs}
    model_inputs = helper.to_device(model_inputs, DEVICE)

    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        result = model.generate_text(
            data=model_inputs,
            temperature=0.6,
            max_generation_length=256,
        )

    print(f"Answer: {result.get('answer', ['N/A'])[0]}")


def main():
    print("Creating dummy data...")
    data = make_dummy_data()
    print(f"  image_frames: {data['image_frames'].shape}")
    print(f"  camera_indices: {data['camera_indices']}")
    print(f"  ego_history_xyz: {data['ego_history_xyz'].shape}")

    print(f"\nLoading model from {MODEL_PATH}...")
    model = load_model()
    print("Model loaded.")

    processor = helper.get_processor(model.tokenizer)
    print("Processor ready.")

    run_vqa_inference(model, processor, data)
    run_trajectory_inference(model, processor, data)

    print("\nDone!")


if __name__ == "__main__":
    main()
