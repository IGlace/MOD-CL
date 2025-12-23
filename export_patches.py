import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from multiprocessing import Pool, cpu_count
from functools import partial

import cv2
import fiftyone as fo
from dotenv import load_dotenv
import numpy as np
from fiftyone.utils.patches import extract_patch as fo_extract_patch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm


def load_main_dataset() -> fo.Dataset:
    load_dotenv()

    container_dataset_path = os.getenv("CONTAINER_DATASET_PATH")
    project_dataset_volume = os.getenv("PROJECT_DATASET_VOLUME")
    fiftyone_main_folder = os.getenv("FIFTYONE_MAIN_FOLDER")

    if not all([container_dataset_path, project_dataset_volume, fiftyone_main_folder]):
        raise RuntimeError(
            "Missing required environment variables: CONTAINER_DATASET_PATH, PROJECT_DATASET_VOLUME, FIFTYONE_MAIN_FOLDER"
        )

    main_dataset_dir = os.path.join(
        container_dataset_path, project_dataset_volume, fiftyone_main_folder
    )
    main_dataset_name = Path(fiftyone_main_folder).name

    dataset = fo.Dataset.from_dir(
        name=main_dataset_name,
        dataset_dir=main_dataset_dir,
        dataset_type=fo.types.FiftyOneDataset,
        persistent=False,
        overwrite=True,
    )
    return dataset


def load_behaviors_mapping(behaviors_path: Path) -> Dict[str, int]:
    with open(behaviors_path, "r") as f:
        mapping: Dict[str, int] = json.load(f)
    return mapping


def ensure_output_dirs(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def get_sample_split(sample: fo.Sample) -> str:
    """Extracts the split tag ('train', 'val', 'test') from a sample."""
    for tag in sample.tags:
        if tag.startswith("split_"):
            return tag.replace("split_", "")
    return "unassigned"

def build_predictions_dict(
    detection_fields: Dict[str, Any], behaviors: List[str]
) -> Dict[str, bool]:
    """Builds a dictionary of behavior predictions from detection fields."""
    predictions: Dict[str, bool] = {}
    for behavior in behaviors:
        predictions[behavior] = bool(detection_fields.get(behavior, False))
    return predictions


def process_image_patches(image_task: Tuple, base_output_dir: Path, expansion: float, behavior_keys: List[str]) -> Tuple[Dict[str, int], List[Dict]]:
    """
    Worker function to process all patches for a single source image.
    Loads the image once, then extracts, saves, and labels all its required patches.
    Returns counts and label data for consolidated JSON output.
    """
    sample_id, sample_filepath, patches_to_extract, split = image_task
    images_dir = base_output_dir / "images"
    local_counts = {b: 0 for b in behavior_keys}
    label_data_list = []  # Collect all label data for this image

    if not os.path.exists(sample_filepath):
        return local_counts, label_data_list

    try:
        # Load image once using OpenCV
        image_np = cv2.imread(sample_filepath)
        if image_np is None:
            return local_counts, label_data_list
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    except Exception:
        return local_counts, label_data_list

    sample_basename = Path(sample_filepath).stem
    image_ext = Path(sample_filepath).suffix or ".jpg"

    for patch_info in patches_to_extract:
        detection_fields = patch_info["detection_fields"]
        bounding_box = patch_info["bounding_box"]

        # Create a temporary fo.Detection for the patch extractor utility
        detection = fo.Detection(bounding_box=bounding_box)

        patch_np = fo_extract_patch(image_np, detection, alpha=expansion)
        patch_np_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)

        patch_id = patch_info["label_id"]
        if patch_id is None:
            bx = bounding_box
            patch_id = f"{int(bx[0]*10000)}_{int(bx[1]*10000)}_{int(bx[2]*10000)}_{int(bx[3]*10000)}"

        patch_filename = f"{sample_basename}_{patch_id}{image_ext}"
        patch_filepath = images_dir / patch_filename
        cv2.imwrite(str(patch_filepath), patch_np_bgr)

        label_payload = {
            "image": patch_filename,
            "split": split,
            "predictions": build_predictions_dict(detection_fields, behavior_keys),
        }
        
        # Collect label data instead of writing individual files
        label_data_list.append(label_payload)

        for behavior in patch_info["active_behaviors"]:
            local_counts[behavior] += 1

    return local_counts, label_data_list


def export_patches(
    output_dir: Path,
    expansion: float,
    behaviors_filter: Optional[List[str]] = None,
    max_per_class: int = 9223372036854775807,
    num_workers: Optional[int] = None,
) -> None:
    dataset = load_main_dataset()
    behaviors_path = Path("behaviors_mapping.json")
    behaviors_mapping = load_behaviors_mapping(behaviors_path)
    behavior_keys = list(behaviors_mapping.keys())

    target_behaviors = [b for b in (behaviors_filter or behavior_keys) if b in behaviors_mapping]
    if not target_behaviors:
        print("No valid behaviors to export.")
        return

    images_dir = ensure_output_dirs(output_dir)
    per_class_counts: Dict[str, int] = {b: 0 for b in target_behaviors}

    # Step 1: Pre-gather all patch information, grouped by source image
    print("Gathering and filtering patch information...")
    image_tasks: Dict[str, Dict[str, Any]] = {}
    patches_view = dataset.to_patches("detections")
    
    # Determine if we should apply special filtering and randomization
    use_limited_splits = max_per_class < 1000
    allowed_splits = {"train", "val"} if use_limited_splits else {"train", "val", "test"}
    
    if use_limited_splits:
        print(f"max_per_class ({max_per_class}) < 1000: limiting to train/val splits with random selection")
    else:
        print(f"max_per_class ({max_per_class}) >= 1000: using all splits with normal iteration")
    
    # Collect all qualifying patches first (for randomization if needed)
    all_qualifying_patches = []
    
    for patch_sample in patches_view.iter_samples(progress=True):
        original_sample = dataset[patch_sample.sample_id]
        sample_split = get_sample_split(original_sample)
        
        # Filter by split if max_per_class < 1000
        if sample_split not in allowed_splits:
            continue
            
        detection = patch_sample.detections

        if detection is None or not hasattr(detection, 'bounding_box'):
            continue

        active_behaviors = [b for b in target_behaviors if detection[b]]
        if not active_behaviors:
            continue
            
        # Store patch info for processing
        all_qualifying_patches.append({
            "patch_sample": patch_sample,
            "original_sample": original_sample,
            "detection": detection,
            "active_behaviors": active_behaviors,
            "sample_split": sample_split
        })
    
    print(f"Found {len(all_qualifying_patches)} qualifying patches")
    
    # Randomize if we're using limited splits
    if use_limited_splits and all_qualifying_patches:
        print("Randomizing patch selection order...")
        random.shuffle(all_qualifying_patches)
    
    # Process patches (now potentially randomized)
    for patch_info in all_qualifying_patches:
        if all(per_class_counts[b] >= max_per_class for b in target_behaviors):
            break
            
        patch_sample = patch_info["patch_sample"]
        original_sample = patch_info["original_sample"]
        detection = patch_info["detection"]
        active_behaviors = patch_info["active_behaviors"]
        sample_split = patch_info["sample_split"]

        if not all(per_class_counts[b] < max_per_class for b in active_behaviors):
            continue

        sample_id = original_sample.id
        if sample_id not in image_tasks:
            image_tasks[sample_id] = {
                "filepath": original_sample.filepath,
                "patches": [],
                "split": sample_split,
            }

        # Store only the necessary serializable data
        image_tasks[sample_id]["patches"].append({
            "label_id": patch_sample["id"],
            "bounding_box": detection.bounding_box,
            "detection_fields": detection.to_dict(),
            "active_behaviors": active_behaviors,
        })

        for b in active_behaviors:
            per_class_counts[b] += 1

    tasks_to_process = [
        (sid, data["filepath"], data["patches"], data["split"]) for sid, data in image_tasks.items()
    ]

    if not tasks_to_process:
        print("No patches found to export.")
        return

    # Step 2: Process images in parallel
    print(f"Exporting patches from {len(tasks_to_process)} images using parallel workers...")
    worker_count = num_workers if num_workers is not None else cpu_count()
    total_counts: Dict[str, int] = {b: 0 for b in behavior_keys}
    all_label_data = []  # Collect all label data from all workers

    worker_func = partial(
        process_image_patches,
        base_output_dir=output_dir,
        expansion=expansion,
        behavior_keys=behavior_keys,
    )

    with Pool(processes=worker_count) as pool:
        # Use tqdm for a progress bar
        results_iterator = pool.imap_unordered(worker_func, tasks_to_process)
        for result_counts, label_data_batch in tqdm(results_iterator, total=len(tasks_to_process)):
            for behavior, count in result_counts.items():
                total_counts[behavior] += count
            # Collect all label data from this batch
            all_label_data.extend(label_data_batch)

    # Step 3: Write consolidated labels JSON file
    print(f"Writing consolidated labels file with {len(all_label_data)} entries...")
    labels_json_path = output_dir / "labels.json"
    with open(labels_json_path, "w", encoding="utf-8") as f:
        json.dump(all_label_data, f, indent=2)
    
    print(f"Consolidated labels saved to: {labels_json_path}")

    # Render rich table summary (after processing all patches)
    console = Console()
    table = Table(title="Exported patches per behavior", title_style="bold white on black")
    table.add_column("Behavior", style="cyan", no_wrap=True)
    table.add_column("Exported", justify="right", style="green")
    limit_label = "Limit"
    remaining_label = "Remaining"
    table.add_column(limit_label, justify="right", style="magenta")
    table.add_column(remaining_label, justify="right", style="yellow")
    table.add_column("Coverage", justify="right", style="blue")

    finite_limit = max_per_class != 9223372036854775807
    total_exported = 0
    for behavior in sorted(target_behaviors):
        count = total_counts.get(behavior, 0)
        total_exported += count
        if finite_limit:
            remaining = max(0, max_per_class - count)
            # Note: The per-class limit is now a pre-filter, not a hard post-export stop.
            # The actual exported count might be slightly different if multiple behaviors are in one patch.
            # This summary reflects the actual files created.
            coverage = f"{(count / max_per_class * 100):.1f}%" if max_per_class > 0 else "0.0%"
            limit_str = str(max_per_class)
            remaining_str = str(remaining)
        else:
            limit_str = "∞"
            remaining_str = "—"
            coverage = "—"
        table.add_row(behavior, str(count), limit_str, remaining_str, coverage)

    table.caption = f"Total exported crops: {total_exported}"
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export detection patches from the main FiftyOne dataset as cropped images "
            "with a consolidated JSON labels file"
        )
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help=(
            "Destination directory. Will create 'images' subfolder and 'labels.json' file"
        ),
    )
    parser.add_argument(
        "--expansion",
        type=float,
        default=0.0,
        help="Relative expansion to apply around each bounding box (default: 0.1)",
    )
    parser.add_argument(
        "--behaviors",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Behavior classes to include. Only detections with at least one of these set to True will be cropped. "
            "Defaults to all behaviors defined in behaviors_mapping.json"
        ),
    )
    parser.add_argument(
        "--max-per-class",
        dest="max_per_class",
        type=int,
        required=False,
        default=9223372036854775807,
        help=(
            "Maximum number of crops to pre-select per behavior class. "
            "This is a pre-filtering step; the final count may vary slightly. Defaults to no limit"
        ),
    )
    parser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes to use. Defaults to all available CPU cores.",
    )
    args = parser.parse_args()

    export_patches(
        Path(args.output_dir),
        args.expansion,
        behaviors_filter=args.behaviors,
        max_per_class=args.max_per_class,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()


