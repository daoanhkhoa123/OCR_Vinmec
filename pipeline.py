import argparse
import gc
import os
import time

import cv2
import torch

from engine.ultils import align_by_hough
from engine.easyocr_reader import crop_by_keywords
from engine.vlm_reader import get_first_info, get_second_info, load_vlm_model
from engine.checkboxes import detect_checkboxes, classify_checkboxes, summarize_checkbox


def run_pipeline(image_path, aligned_image_path):
    pipeline_start = time.perf_counter()

    # Step 1: Align Image
    t0 = time.perf_counter()
    rotated, angle = align_by_hough(cv2.imread(image_path))
    print(f"Rotation angle: {angle} | Time: {time.perf_counter() - t0:.3f}s")

    cv2.imwrite(aligned_image_path, rotated)
    print(f"Saved to {aligned_image_path}")

    # Step 2: Crop by Keywords
    t0 = time.perf_counter()
    img = cv2.imread(aligned_image_path)
    cropped = crop_by_keywords(img)
    print(f"Crop by keywords | Time: {time.perf_counter() - t0:.3f}s")

    # Step 3: VLM Extraction
    t0 = time.perf_counter()
    first_info = get_first_info(cropped)
    print(first_info)
    print(f"VLM First Info | Time: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    second_info = get_second_info(cropped)
    print(second_info)
    print(f"VLM Second Info | Time: {time.perf_counter() - t0:.3f}s")

    # Step 4: Checkbox Detection & Classification
    t0 = time.perf_counter()
    stats, labels = detect_checkboxes(cropped)
    checkbox_dict = classify_checkboxes(cropped, stats, n_groups=13)
    checkbox_info = summarize_checkbox(checkbox_dict)
    print(checkbox_info)
    print(f"Checkbox Processing | Time: {time.perf_counter() - t0:.3f}s")

    # Final Summary
    info = {**first_info, **second_info, **checkbox_info}
    print("\nMerged Output:", info)

    pipeline_duration = time.perf_counter() - pipeline_start
    print(f"\n==========================================")
    print(f"Total Pipeline Execution Time: {pipeline_duration:.3f}s")
    print(f"==========================================")

    return info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Vinmec eye-exam OCR pipeline on one or more scanned form images."
    )
    parser.add_argument("images", nargs="+", help="Path(s) to the input form image(s).")
    parser.add_argument(
        "--aligned-suffix", default="_aligned",
        help="Suffix appended to each input filename for its deskewed copy (default: _aligned).",
    )
    parser.add_argument(
        "--device", default=None, choices=["cuda", "cpu"],
        help="Force the VLM onto this device instead of auto placement. Use 'cpu' if you hit CUDA OOM.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load the (large) VLM once per process - re-running this script per image
    # reloads an 8B-parameter model from scratch every time, which is the
    # usual cause of CUDA OOM on a second invocation before the first
    # process's GPU memory has been freed. Passing multiple images to a
    # single run keeps the model loaded once and reused.
    load_vlm_model(device=args.device)

    for image_path in args.images:
        stem, ext = os.path.splitext(image_path)
        aligned_image_path = f"{stem}{args.aligned_suffix}{ext or '.png'}"
        try:
            run_pipeline(image_path, aligned_image_path)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
