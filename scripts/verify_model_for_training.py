#!/usr/bin/env python3
"""
Verify model is compatible with prime-rl training before deploying.

Run this BEFORE creating a GPU pod to catch compatibility issues early.

Usage:
    python scripts/verify_model_for_training.py Qwen/Qwen2.5-7B-Instruct
    python scripts/verify_model_for_training.py mistralai/Ministral-3-8B-Instruct-2512

Checks:
    1. Model exists on HuggingFace Hub
    2. Model is NOT a Vision-Language model (VL models need different handling)
    3. Transformers can load the model config
    4. Model type is registered in transformers (no KeyError)

Known blocked models:
    - gpt-oss-20b: vLLM weight reload bug (#28606)
    - qwen3-vl-*: VL model, wrong class
    - Ministral-3-8B: transformers KeyError
"""

import argparse
import sys

# Known blocked models with reasons
BLOCKED_MODELS = {
    "openai/gpt-oss-20b": "vLLM weight reload bug (TypeError: default_weight_loader)",
    "mistralai/Ministral-3-8B-Instruct-2512": "transformers KeyError: 'ministral3'",
}

# Keywords that indicate Vision-Language models
VL_KEYWORDS = ["vl", "vision", "multimodal", "image", "visual"]


def check_model(model_name: str, verbose: bool = True) -> list[str]:
    """Return list of issues with the model."""
    issues = []

    # Check if explicitly blocked
    if model_name in BLOCKED_MODELS:
        issues.append(f"Model is BLOCKED: {BLOCKED_MODELS[model_name]}")
        return issues  # No point checking further

    # Check for VL/Vision models by name
    model_lower = model_name.lower()
    vl_matches = [kw for kw in VL_KEYWORDS if kw in model_lower]
    if vl_matches:
        issues.append(
            f"Model appears to be Vision-Language (name contains: {vl_matches}). "
            "VL models require AutoModelForVision2Seq, not AutoModelForCausalLM."
        )

    # Check HuggingFace availability
    try:
        from huggingface_hub import model_info

        info = model_info(model_name)
        if verbose:
            print(f"✓ Model found on HuggingFace: {info.modelId}")

        # Check model tags for vision
        if info.tags:
            vision_tags = [t for t in info.tags if any(kw in t.lower() for kw in VL_KEYWORDS)]
            if vision_tags:
                issues.append(f"Model has vision-related tags: {vision_tags}")

    except Exception as e:
        issues.append(f"Model not found on HuggingFace: {e}")

    # Check transformers compatibility
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if verbose:
            print(f"✓ Transformers config loaded: {config.model_type}")

        # Check for vision-related config attributes
        if hasattr(config, "vision_config") or hasattr(config, "image_size"):
            issues.append(
                "Model config has vision-related attributes (vision_config, image_size). "
                "This is likely a VL model."
            )

    except KeyError as e:
        issues.append(
            f"Transformers KeyError loading config: {e}. "
            "Model type not registered in transformers library."
        )
    except Exception as e:
        issues.append(f"Transformers cannot load config: {e}")

    # Try loading tokenizer (catches some issues)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if verbose:
            print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")
    except Exception as e:
        # Tokenizer issues are warnings, not blockers
        if verbose:
            print(f"⚠ Tokenizer warning: {e}")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Verify model is compatible with prime-rl training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", help="HuggingFace model name to verify")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only output errors")
    args = parser.parse_args()

    print(f"Checking model: {args.model}\n")

    issues = check_model(args.model, verbose=not args.quiet)

    if issues:
        print(f"\n❌ {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nModel may NOT be compatible with prime-rl training.")
        sys.exit(1)
    else:
        print("\n✓ Model appears compatible with prime-rl training")
        print("\nRecommended next steps:")
        print("  1. Check if model supports tool calling (required for BS4 env)")
        print("  2. Run baseline benchmark if possible")
        print("  3. Create pod and test training with max_steps=5")
        sys.exit(0)


if __name__ == "__main__":
    main()
