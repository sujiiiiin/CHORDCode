import argparse
from pathlib import Path

import torch

from prompting_utils import (
    VLMClient,
    add_render_args,
    add_scene_args,
    add_vlm_args,
    build_prompt_suggestion_instruction,
    ensure_dir,
    parse_prompt_list,
    render_multiview_gaussian_stills,
    write_text,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Suggest feasible CHORD motion prompts from multiview Gaussian renders."
    )
    lp = add_scene_args(parser)
    add_render_args(parser)
    add_vlm_args(parser)
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()
    dataset = lp.extract(args)
    return args, dataset


def main():
    args, dataset = parse_args()
    output_dir = Path(args.output_dir)
    render_dir = output_dir / "renders"
    ensure_dir(output_dir)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    view_records = render_multiview_gaussian_stills(args, dataset, render_dir)
    image_paths = [Path(record["image"]) for record in view_records]

    txt_path = output_dir / "prompt_suggestions.txt"
    if txt_path.is_file() and not args.overwrite:
        print(f"[PromptTools] Existing suggestions found: {txt_path}")
        return

    client = VLMClient(args)
    instruction = build_prompt_suggestion_instruction(args.num_prompts)
    raw_text = client.generate_json(instruction, image_paths=image_paths)
    prompts = parse_prompt_list(raw_text)[: args.num_prompts]

    write_text(txt_path, "\n".join(prompts) + "\n")
    print(f"[PromptTools] Wrote {len(prompts)} prompt suggestions to {txt_path}")


if __name__ == "__main__":
    main()
