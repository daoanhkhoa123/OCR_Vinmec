import re

import cv2
import numpy as np
import matplotlib.pyplot as plt


def align_by_hough(cropped, canny_low=50, canny_high=150, hough_thresh=120):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh)
    best_angle = 0.0

    if lines is not None and len(lines) > 0:
        rho, theta = lines[0][0]

        # In OpenCV, theta=0 is vertical; rotate by (theta - π/2) to make it horizontal
        best_angle = np.degrees(theta - np.pi / 2)

    # --- expand canvas to avoid black corners ---
    h, w = cropped.shape[:2]
    new_size = int(max(h, w) * 1.5)
    canvas = np.full((new_size, new_size, 3), 255, dtype=np.uint8)

    x_offset = (new_size - w) // 2
    y_offset = (new_size - h) // 2
    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = cropped

    # --- rotate around canvas center ---
    center = (new_size // 2, new_size // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(canvas, M, (new_size, new_size), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    # --- crop back to original size ---
    rotated = rotated[y_offset:y_offset + h, x_offset:x_offset + w]

    return rotated, best_angle


def extract_field(text: str, field_name: str) -> str | None:
    pattern = rf"{re.escape(field_name)}\s*[:：]\s*(.*)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip().rstrip(",") or None
    return None


def extract_fields(text: str, field_names) -> dict:
    res = {}
    for fn in field_names:
        val = extract_field(text, fn)
        res[fn] = val.replace("*", "").strip() if val else val

    return res


def parse_json_like(s: str) -> dict:
    def parse_block(block: str) -> dict:
        pairs = re.findall(r'"([^"]+)"\s*:\s*(\{[^{}]*\}|"[^"]*")', block, re.DOTALL)
        result = {}
        for key, val in pairs:
            val = val.strip()
            if val.startswith("{") and val.endswith("}"):
                result[key] = parse_block(val[1:-1])
            else:
                m = re.match(r'"(.*)"', val)
                result[key] = m.group(1).strip() if m else val
        return result

    return parse_block(s.strip().strip("{}"))


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    annotated = image.copy()
    for b in boxes:
        pts = np.array(b["bbox"], np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], True, color, thickness)
        cv2.putText(
            annotated, b["text"],
            tuple(pts[0][0]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 0, 0), 2
        )
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def plot_image(image, title='Result'):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
