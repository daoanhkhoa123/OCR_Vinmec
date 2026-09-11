import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .easyocr_reader import get_reader, crop_by_keywords
from .ultils import extract_fields, draw_boxes

_model = None
_processor = None
_device = None


def load_vlm_model(model_name="Qwen/Qwen3-VL-8B-Instruct", device=None):
    global _model, _processor, _device
    if _model is None or _processor is None:
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map=(device or "auto")
        )
        _processor = AutoProcessor.from_pretrained(model_name)
    return _model, _processor, _device


def process_image(image, user_request, model=None, processor=None, device=None, max_new_tokens=256):
    """
    Xử lý ảnh và chạy qua mô hình Qwen3-VL.
    """
    if model is None or processor is None:
        model, processor, default_device = load_vlm_model()
        device = device or default_device

    if not user_request or not user_request.strip():
        user_request = "Trích xuất toàn bộ thông tin trong ảnh và trả về dạng Markdown."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_request},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


def extract_patch(img, start, end, all_line_ys, show=False):
    y1 = all_line_ys[start]
    y2 = all_line_ys[end]
    patch = img[y1:y2, :]
    if show:
        pass  # local run: skip cv2_imshow (Colab-only), visualization disabled
    return patch


def read_patch(img, *, user_request=None, model=None, processor=None, device=None, show=False):
    image = Image.fromarray(img)
    text = process_image(image, user_request, model, processor, device)
    return text


def get_first_info(img, debug=False):
    crop = img.copy()
    crop = crop_by_keywords(crop, bottom_words="điều trị kiểm ", debug=debug)
    first_text = read_patch(crop, show=debug)
    d1 = extract_fields(first_text, ["Họ và tên", "Ngày/tháng/năm sinh", "Lớp"])
    return d1


def get_second_info(img, debug=False):
    # Extract region for vision test
    crop = img.copy()
    crop = crop_by_keywords(crop, "Mắt Phải", "Không đạt", c=5, debug=debug)
    reader = get_reader()
    results = reader.readtext(crop)

    boxes = []
    number_reading = []

    for bbox, text, conf in results:
        if "10" in text:  # crude filter, adjust if needed
            boxes.append({
                "bbox": np.array(bbox).astype(int).tolist(),
                "text": text,
                "conf": float(conf)
            })

            # Crop region
            x_min = int(min([p[0] for p in bbox]))
            y_min = int(min([p[1] for p in bbox]))
            x_max = int(max([p[0] for p in bbox]))
            y_max = int(max([p[1] for p in bbox]))
            region = crop[y_min:y_max, x_min:x_max]

            region_pil = Image.fromarray(region)

            # OCR with Qwen3-VL
            ocr_text = process_image(
                region_pil,
                user_request="Chỉ trích xuất số trong vùng này.",
            )

            # Clean OCR result
            tokens = ocr_text.split()
            if tokens:
                clean_text = "".join(tokens[:len(tokens) // 2])
                number_reading.append(clean_text + "/10")

    # Debug visualization
    if debug:
        patch = crop.copy()
        draw_boxes(patch, boxes)

    # Safety check: must have 4 values
    if len(number_reading) < 4:
        # Fill missing with None or "N/A"
        number_reading += ["N/A"] * (4 - len(number_reading))

    vision_dict = {
        "Thị lực nhìn xa không kính": {
            "Mắt Phải": number_reading[0],
            "Mắt Trái": number_reading[1]
        },
        "Thị lực nhìn xa có kính": {
            "Mắt Phải": number_reading[2],
            "Mắt Trái": number_reading[3]
        }
    }

    return vision_dict
