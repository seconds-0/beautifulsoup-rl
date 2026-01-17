#!/usr/bin/env python3
"""Convert a distributed checkpoint to PEFT adapter format for resume."""

import json
import os

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load


def convert_checkpoint(ckpt_dir: str, broadcast_dir: str):
    os.makedirs(broadcast_dir, exist_ok=True)

    print(f"Loading checkpoint from {ckpt_dir}...")

    # Load the checkpoint using DCP
    reader = FileSystemReader(ckpt_dir)
    metadata = reader.read_metadata()

    # Get all keys that are LoRA related
    lora_keys = [k for k in metadata.state_dict_metadata if "lora_" in k]
    print(f"Found {len(lora_keys)} LoRA keys")

    # Create placeholder tensors for LoRA keys only
    state_dict = {}
    for key in lora_keys:
        tensor_meta = metadata.state_dict_metadata[key]
        state_dict[key] = torch.zeros(tensor_meta.size, dtype=tensor_meta.properties.dtype)

    # Load the state dict
    load(state_dict=state_dict, storage_reader=reader)
    print(f"Loaded {len(state_dict)} LoRA tensors")

    # Extract and rename LoRA weights to PEFT format
    peft_state_dict = {}
    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            # Convert from: app.model.model.layers.0.self_attn.q_proj.lora_A.0
            # To: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
            new_key = key.replace("app.model.", "base_model.model.")
            new_key = new_key.replace(".lora_A.0", ".lora_A.weight")
            new_key = new_key.replace(".lora_B.0", ".lora_B.weight")
            peft_state_dict[new_key] = value

    print(f"Extracted {len(peft_state_dict)} LoRA weights")

    # Save as safetensors
    save_file(peft_state_dict, os.path.join(broadcast_dir, "adapter_model.safetensors"))
    print("Saved adapter_model.safetensors")

    # Create adapter_config.json
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "Qwen/Qwen3-8B",
        "r": 8,
        "lora_alpha": 32.0,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": [
            "down_proj",
            "gate_proj",
            "k_proj",
            "o_proj",
            "q_proj",
            "up_proj",
            "v_proj",
        ],
        "modules_to_save": None,
    }
    with open(os.path.join(broadcast_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("Saved adapter_config.json")

    # Create STABLE marker
    open(os.path.join(broadcast_dir, "STABLE"), "w").close()
    print("Created STABLE marker")

    print(f"Broadcast directory ready: {os.listdir(broadcast_dir)}")


if __name__ == "__main__":
    import sys

    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/outputs/checkpoints/step_775/trainer"
    broadcast_dir = (
        sys.argv[2] if len(sys.argv) > 2 else "/app/outputs/run_default/broadcasts/step_775"
    )
    convert_checkpoint(ckpt_dir, broadcast_dir)
