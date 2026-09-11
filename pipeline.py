import cv2

from engine.ultils import align_by_hough
from engine.easyocr_reader import crop_by_keywords
from engine.vlm_reader import get_first_info, get_second_info
from engine.checkboxes import detect_checkboxes, classify_checkboxes, summarize_checkbox

IMAGE_PATH = "handwriting_test.png"
ALIGNED_IMAGE_PATH = "handwriting_aligned.png"

if __name__ == "__main__":
    rotated, angle = align_by_hough(cv2.imread(IMAGE_PATH))
    print("rotation angle:", angle)
    cv2.imwrite(ALIGNED_IMAGE_PATH, rotated)
    print(f"saved to {ALIGNED_IMAGE_PATH}")

    img = cv2.imread(ALIGNED_IMAGE_PATH)
    cropped = crop_by_keywords(img)

    first_info = get_first_info(cropped)
    print(first_info)

    second_info = get_second_info(cropped)
    print(second_info)

    stats, labels = detect_checkboxes(cropped)
    checkbox_dict = classify_checkboxes(cropped, stats, n_groups=13)
    checkbox_info = summarize_checkbox(checkbox_dict)
    print(checkbox_info)

    info = {**first_info, **second_info, **checkbox_info}
    print(info)
