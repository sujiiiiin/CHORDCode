import argparse
import json
import shutil
from pathlib import Path

import torch

from prompting_utils import (
    VLMClient,
    add_render_args,
    add_scene_args,
    add_vlm_args,
    add_video_generation_args,
    apply_video_render_size,
    build_prompt_expansion_instruction,
    build_target_requirements_instruction,
    build_video_judgment_instruction,
    create_video_pipeline,
    ensure_dir,
    generate_prompt_video,
    parse_judgment,
    parse_prompt_list,
    parse_requirement_list,
    render_multiview_gaussian_stills,
    resolve_video_generation_args,
    sanitize_name,
    write_json,
    write_text,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate whether a motion prompt works with the video generation model."
    )
    lp = add_scene_args(parser)
    add_render_args(parser)
    add_vlm_args(parser)
    add_video_generation_args(parser)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--judge", choices=["manual", "vlm"], default="manual")
    parser.add_argument("--expand_prompts", action="store_true", default=False)
    parser.add_argument("--num_prompt_expansions", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--refresh_judgments", action="store_true", default=False)
    args = parser.parse_args()
    resolve_video_generation_args(args)
    dataset = lp.extract(args)
    apply_video_render_size(args, dataset)
    return args, dataset


def unique_prompt_candidates(generation_prompts, target_prompt: str):
    seen = set()
    candidates = []
    for prompt_index, prompt in enumerate(generation_prompts):
        key = prompt.lower().strip()
        if key and key not in seen:
            seen.add(key)
            candidates.append(
                {
                    "prompt_index": len(candidates),
                    "target_prompt": target_prompt,
                    "generation_prompt": prompt.strip(),
                    "source": "original" if prompt_index == 0 else "expansion",
                }
            )
    return candidates


def collect_prompt_candidates(args, dataset, view_records, output_dir: Path):
    generation_prompts = [args.prompt]
    raw_expansion_path = output_dir / "vlm" / "prompt_expansions_raw.txt"
    expansion_json_path = output_dir / "prompt_expansions.json"
    if args.expand_prompts:
        if expansion_json_path.is_file() and not args.overwrite:
            payload = json.loads(expansion_json_path.read_text())
            cached_candidates = payload.get("prompt_candidates", [])
            if cached_candidates:
                return cached_candidates
            cached_prompts = payload.get("generation_prompts", [])
            if cached_prompts:
                return unique_prompt_candidates(cached_prompts, args.prompt)

        client = VLMClient(args)
        image_paths = [Path(record["image"]) for record in view_records]
        instruction = build_prompt_expansion_instruction(args.prompt, args.num_prompt_expansions)
        raw_text = client.generate_json(instruction, image_paths=image_paths)
        expansions = parse_prompt_list(raw_text)[: args.num_prompt_expansions]
        generation_prompts.extend(expansions)
        candidates = unique_prompt_candidates(generation_prompts, args.prompt)
        write_text(raw_expansion_path, raw_text + "\n")
        write_json(
            expansion_json_path,
            {
                "model_path": dataset.model_path,
                "mesh_source_path": dataset.mesh_source_path,
                "target_prompt": args.prompt,
                "generation_prompts": [candidate["generation_prompt"] for candidate in candidates],
                "prompt_candidates": candidates,
                "vlm": args.vlm,
            },
        )
        return candidates
    return unique_prompt_candidates(generation_prompts, args.prompt)


def clean_output_dir(output_dir: Path) -> None:
    for rel_path in [
        "renders",
        "videos",
        "video_validation",
        "vlm",
    ]:
        path = output_dir / rel_path
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for rel_path in [
        "evaluation_summary.json",
        "final_verdict.json",
        "final_verdict.txt",
        "prompt_expansions.json",
        "prompt_rewrites.json",
        "target_requirements.json",
    ]:
        (output_dir / rel_path).unlink(missing_ok=True)


def collect_target_requirements(args, client, output_dir: Path):
    raw_requirement_path = output_dir / "vlm" / "target_requirements_raw.txt"

    raw_text = client.generate_json(build_target_requirements_instruction(args.prompt))
    requirements = parse_requirement_list(raw_text)
    write_text(raw_requirement_path, raw_text + "\n")
    return requirements


def evaluate_one_video(
    args,
    client,
    target_prompt: str,
    target_requirements,
    video_path: Path,
    output_json: Path,
    raw_path: Path,
):
    if output_json.is_file() and not args.overwrite and not args.refresh_judgments:
        cached = json.loads(output_json.read_text())
        if {
            "motion_adherence_score",
            "video_quality_score",
            "combined_score",
            "raw_judgment",
        }.issubset(cached):
            return cached
        return parse_judgment(json.dumps(cached))
    raw_text = client.generate_json(
        build_video_judgment_instruction(target_prompt, target_requirements),
        video_path=video_path,
    )
    judgment = parse_judgment(raw_text)
    write_text(raw_path, raw_text + "\n")
    write_json(output_json, judgment)
    return judgment


def average_score(records, key):
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def summarize_prompt_record(record):
    evaluated_count = int(record.get("evaluated_count", 0))
    if evaluated_count <= 0:
        return {
            "summary": "manual_review_required",
            "reason": "No VLM judgments were requested for this prompt.",
        }

    return {
        "summary": "scored",
        "reason": (
            f"Average scores over {evaluated_count} views: "
            f"motion={record['average_motion_adherence_score']:.3f}, "
            f"quality={record['average_video_quality_score']:.3f}, "
            f"combined={record['average_combined_score']:.3f}."
        ),
    }


def prompt_verdict_summary(record):
    return {
        "prompt_index": record["prompt_index"],
        "prompt_id": record["prompt_id"],
        "source": record["source"],
        "target_prompt": record["target_prompt"],
        "generation_prompt": record["generation_prompt"],
        "evaluated_count": record["evaluated_count"],
        "average_motion_adherence_score": record["average_motion_adherence_score"],
        "average_video_quality_score": record["average_video_quality_score"],
        "average_combined_score": record["average_combined_score"],
        "summary": record["score_summary"],
        "reason": record["score_reason"],
    }


def build_final_verdict(args, prompt_records):
    prompt_summaries = [prompt_verdict_summary(record) for record in prompt_records]
    if args.judge != "vlm":
        return {
            "summary": "manual_review_required",
            "reason": "Videos were generated for manual review, but no VLM judgments were requested.",
            "judge": args.judge,
            "target_prompt": args.prompt,
            "highest_scoring_generation_prompt": None,
            "overall_motion_adherence_score": None,
            "overall_video_quality_score": None,
            "overall_combined_score": None,
            "prompt_verdicts": prompt_summaries,
        }

    all_judgments = [
        video["judgment"]
        for record in prompt_records
        for video in record["videos"]
        if "judgment" in video
    ]
    best_record = max(
        prompt_records,
        key=lambda record: (
            float(record["average_combined_score"] or 0.0),
            float(record["average_motion_adherence_score"] or 0.0),
            float(record["average_video_quality_score"] or 0.0),
            -int(record["prompt_index"]),
        ),
    )
    best_summary = prompt_verdict_summary(best_record)
    reason = (
        "Highest-scoring generation prompt has "
        f"combined={best_summary['average_combined_score']:.3f}, "
        f"motion={best_summary['average_motion_adherence_score']:.3f}, "
        f"quality={best_summary['average_video_quality_score']:.3f}."
    )

    return {
        "summary": "scored",
        "reason": reason,
        "judge": args.judge,
        "target_prompt": args.prompt,
        "highest_scoring_generation_prompt": best_summary,
        "overall_motion_adherence_score": average_score(all_judgments, "motion_adherence_score"),
        "overall_video_quality_score": average_score(all_judgments, "video_quality_score"),
        "overall_combined_score": average_score(all_judgments, "combined_score"),
        "prompt_verdicts": prompt_summaries,
    }


def format_final_verdict(final_verdict):
    lines = [
        f"summary: {final_verdict['summary']}",
        f"reason: {final_verdict['reason']}",
        f"target_prompt: {final_verdict['target_prompt']}",
    ]
    if final_verdict["overall_combined_score"] is not None:
        lines.extend(
            [
                f"overall_motion_adherence_score: {final_verdict['overall_motion_adherence_score']:.4f}",
                f"overall_video_quality_score: {final_verdict['overall_video_quality_score']:.4f}",
                f"overall_combined_score: {final_verdict['overall_combined_score']:.4f}",
            ]
        )
    if final_verdict["highest_scoring_generation_prompt"] is not None:
        best = final_verdict["highest_scoring_generation_prompt"]
        lines.extend(
            [
                f"highest_scoring_generation_prompt_index: {best['prompt_index']}",
                f"highest_scoring_generation_prompt_motion_score: {best['average_motion_adherence_score']:.4f}",
                f"highest_scoring_generation_prompt_quality_score: {best['average_video_quality_score']:.4f}",
                f"highest_scoring_generation_prompt_combined_score: {best['average_combined_score']:.4f}",
                f"highest_scoring_generation_prompt: {best['generation_prompt']}",
            ]
        )
    if final_verdict["prompt_verdicts"]:
        lines.append("generation_prompt_scores:")
        for prompt in final_verdict["prompt_verdicts"]:
            motion_score = prompt["average_motion_adherence_score"]
            quality_score = prompt["average_video_quality_score"]
            combined_score = prompt["average_combined_score"]
            if combined_score is None:
                score_text = "motion=n/a quality=n/a combined=n/a"
            else:
                score_text = (
                    f"motion={motion_score:.4f} "
                    f"quality={quality_score:.4f} "
                    f"combined={combined_score:.4f}"
                )
            lines.append(
                "  "
                f"[{prompt['prompt_index']}] {prompt['evaluated_count']} views "
                f"{score_text}: {prompt['generation_prompt']}"
            )
    return "\n".join(lines) + "\n"


def print_final_verdict(final_verdict):
    print(f"[PromptTools] Final scoring summary: {final_verdict['summary']} - {final_verdict['reason']}")
    print(f"[PromptTools] Target prompt: {final_verdict['target_prompt']}")
    best = final_verdict["highest_scoring_generation_prompt"]
    if best is not None:
        print(
            "[PromptTools] Highest-scoring generation prompt "
            f"[{best['prompt_index']}]: "
            f"motion={best['average_motion_adherence_score']:.3f}, "
            f"quality={best['average_video_quality_score']:.3f}, "
            f"combined={best['average_combined_score']:.3f} - "
            f"{best['generation_prompt']}"
        )
    if final_verdict["prompt_verdicts"]:
        print("[PromptTools] Generation prompt scores:")
        for prompt in final_verdict["prompt_verdicts"]:
            if prompt["average_combined_score"] is None:
                score_text = "motion=n/a quality=n/a combined=n/a"
            else:
                score_text = (
                    f"motion={prompt['average_motion_adherence_score']:.3f}, "
                    f"quality={prompt['average_video_quality_score']:.3f}, "
                    f"combined={prompt['average_combined_score']:.3f}"
                )
            print(
                "[PromptTools]   "
                f"[{prompt['prompt_index']}] {prompt['evaluated_count']} views "
                f"{score_text} - {prompt['generation_prompt']}"
            )


def main():
    args, dataset = parse_args()
    output_dir = Path(args.output_dir)
    render_dir = output_dir / "renders"
    video_root = output_dir / "videos"
    validation_root = output_dir / "video_validation"
    if args.overwrite and output_dir.exists():
        clean_output_dir(output_dir)
    ensure_dir(output_dir)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    view_records = render_multiview_gaussian_stills(args, dataset, render_dir)
    prompt_candidates = collect_prompt_candidates(args, dataset, view_records, output_dir)

    judge_client = VLMClient(args) if args.judge == "vlm" else None
    target_requirements = (
        collect_target_requirements(args, judge_client, output_dir)
        if args.judge == "vlm"
        else []
    )
    video_pipeline = None
    cfg = None
    prompt_records = []

    for candidate in prompt_candidates:
        prompt_index = candidate["prompt_index"]
        target_prompt = candidate["target_prompt"]
        generation_prompt = candidate["generation_prompt"]
        prompt_id = f"prompt_{prompt_index:03d}_{sanitize_name(generation_prompt, 60)}"
        cur_video_dir = video_root / prompt_id
        cur_validation_dir = validation_root / prompt_id
        ensure_dir(cur_video_dir)
        ensure_dir(cur_validation_dir)
        record = {
            "prompt_index": prompt_index,
            "prompt_id": prompt_id,
            "source": candidate["source"],
            "target_prompt": target_prompt,
            "generation_prompt": generation_prompt,
            "videos": [],
            "evaluated_count": 0,
            "average_motion_adherence_score": None,
            "average_video_quality_score": None,
            "average_combined_score": None,
        }

        for view in view_records:
            image_path = Path(view["image"])
            view_name = image_path.stem
            video_path = cur_video_dir / f"{view_name}.mp4"
            if not video_path.is_file() or args.overwrite:
                print(f"[PromptTools] Generating video prompt={prompt_id} view={view_name}")
                if video_pipeline is None:
                    video_pipeline, cfg = create_video_pipeline(args)
                generate_prompt_video(video_pipeline, cfg, image_path, generation_prompt, video_path, args)

            video_record = {
                "view": view,
                "video": str(video_path),
            }
            if args.judge == "vlm":
                print(f"[PromptTools] Judging video prompt={prompt_id} view={view_name}")
                judgment = evaluate_one_video(
                    args,
                    judge_client,
                    target_prompt,
                    target_requirements,
                    video_path,
                    cur_validation_dir / f"{view_name}.json",
                    cur_validation_dir / f"{view_name}_raw.txt",
                )
                video_record["judgment"] = judgment
                record["evaluated_count"] += 1

            record["videos"].append(video_record)

        if args.judge == "vlm":
            judgments = [video["judgment"] for video in record["videos"] if "judgment" in video]
            record["average_motion_adherence_score"] = average_score(judgments, "motion_adherence_score")
            record["average_video_quality_score"] = average_score(judgments, "video_quality_score")
            record["average_combined_score"] = average_score(judgments, "combined_score")
        prompt_summary = summarize_prompt_record(record)
        record["score_summary"] = prompt_summary["summary"]
        record["score_reason"] = prompt_summary["reason"]
        prompt_records.append(record)

    final_verdict = build_final_verdict(args, prompt_records)
    write_json(output_dir / "final_verdict.json", final_verdict)
    write_text(output_dir / "final_verdict.txt", format_final_verdict(final_verdict))
    print_final_verdict(final_verdict)
    print(f"[PromptTools] Wrote final scoring summary to {output_dir / 'final_verdict.json'}")


if __name__ == "__main__":
    main()
