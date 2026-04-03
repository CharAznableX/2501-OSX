#!/usr/bin/env python3
"""
Post-install patches for vmlx-engine dependencies.

Fixes issues in transformers, mlx-vlm, and mlx-lm when running in
a torch-free environment (no PyTorch, no torchvision, no soundfile).

Usage:
    python3 post_install_patches.py <site-packages-path>

Each patch checks whether the target code still matches before modifying,
so running this script multiple times is safe.
"""

import glob
import os
import sys


def patch_file(path, old, new, description):
    """Replace old with new in file at path. Returns True if patched."""
    if not os.path.isfile(path):
        print(f"  Skipped: {os.path.basename(path)} not found")
        return False
    with open(path, "r") as f:
        content = f.read()
    if old not in content:
        print(f"  Already patched or structure changed: {description}")
        return False
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print(f"  Patched: {description}")
    return True


def patch_1_none_sub_processors(site):
    """transformers/processing_utils.py: Allow None sub-processors.
    Without torchvision, Qwen2VL's video_processor loads as None."""
    path = os.path.join(site, "transformers", "processing_utils.py")
    patch_file(
        path,
        "if not isinstance(argument, proper_class):",
        "if argument is not None and not isinstance(argument, proper_class):",
        "processing_utils.py None sub-processor guard",
    )


def patch_2_import_error_sub_processors(site):
    """transformers/processing_utils.py: Skip ImportError when loading
    sub-processors that require torchvision."""
    path = os.path.join(site, "transformers", "processing_utils.py")
    old = (
        "            elif is_primary:\n"
        "                # Primary non-tokenizer sub-processor: load via Auto class\n"
        "                auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[sub_processor_type]\n"
        "                sub_processor = auto_processor_class.from_pretrained(\n"
        "                    pretrained_model_name_or_path, subfolder=subfolder, **kwargs\n"
        "                )\n"
        "                args.append(sub_processor)"
    )
    new = (
        "            elif is_primary:\n"
        "                # Primary non-tokenizer sub-processor: load via Auto class\n"
        "                auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[sub_processor_type]\n"
        "                try:\n"
        "                    sub_processor = auto_processor_class.from_pretrained(\n"
        "                        pretrained_model_name_or_path, subfolder=subfolder, **kwargs\n"
        "                    )\n"
        "                    args.append(sub_processor)\n"
        "                except ImportError:\n"
        "                    pass"
    )
    patch_file(path, old, new, "processing_utils.py sub-processor ImportError handling")


def patch_3_video_processing_null_check(site):
    """transformers/models/auto/video_processing_auto.py: Null check for
    extractors (transformers 5.2.0 bug where extractors can be None)."""
    path = os.path.join(
        site, "transformers", "models", "auto", "video_processing_auto.py"
    )
    patch_file(
        path,
        "if class_name in extractors:",
        "if extractors is not None and class_name in extractors:",
        "video_processing_auto.py null extractors guard",
    )


def patch_4_lazy_soundfile(site):
    """mlx_vlm/utils.py: Lazy-import soundfile (not bundled)."""
    path = os.path.join(site, "mlx_vlm", "utils.py")
    patch_file(
        path,
        "import soundfile as sf",
        "# import soundfile as sf  # lazy-loaded: see _get_sf()",
        "mlx_vlm/utils.py lazy soundfile import",
    )


def patch_5_qwen35_mrope(site):
    """mlx_vlm/models/qwen3_5/language.py: Fix mRoPE dimension mismatch
    for MoE models (mlx-vlm 0.3.12 bug)."""
    path = os.path.join(site, "mlx_vlm", "models", "qwen3_5", "language.py")
    old = (
        "    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)\n"
        "    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)\n"
        "\n"
        "    q_embed = mx.concatenate([q_embed, q_pass], axis=-1)"
    )
    new = (
        "    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)\n"
        "    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)\n"
        "\n"
        "    if q_embed.ndim > q_pass.ndim and q_embed.ndim == 5:\n"
        "        q_embed = q_embed[0]\n"
        "        k_embed = k_embed[0]\n"
        "\n"
        "    q_embed = mx.concatenate([q_embed, q_pass], axis=-1)"
    )
    patch_file(path, old, new, "qwen3_5/language.py mRoPE dimension fix")


def patch_6_ssm_mamba(site):
    """mlx_lm/models/ssm.py: Mamba state fixes (dt clip and state dtype)."""
    paths = glob.glob(os.path.join(site, "mlx_lm", "models", "ssm.py"))
    if not paths:
        print("  Skipped: ssm.py not found")
        return
    path = paths[0]
    patch_file(
        path,
        "return mx.clip(dt, time_step_limit[0], time_step_limit[1])",
        "return mx.maximum(dt, time_step_limit[0])",
        "ssm.py dt clip -> maximum",
    )
    patch_file(
        path,
        "output_dtypes=[input_type, input_type]",
        "output_dtypes=[input_type, mx.float32]",
        "ssm.py state dtype -> float32",
    )


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <site-packages-path>")
        sys.exit(1)

    site = sys.argv[1]
    if not os.path.isdir(site):
        print(f"Error: {site} is not a directory")
        sys.exit(1)

    print(f"Applying post-install patches to {site}")
    patch_1_none_sub_processors(site)
    patch_2_import_error_sub_processors(site)
    patch_3_video_processing_null_check(site)
    patch_4_lazy_soundfile(site)
    patch_5_qwen35_mrope(site)
    patch_6_ssm_mamba(site)
    print("All patches applied.")


if __name__ == "__main__":
    main()
