import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from .easyocr_reader import get_reader, crop_by_keywords
from .ultils import extract_fields, draw_boxes

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_model = None
_tokenizer = None
_device = None


def load_vlm_model(model_name="5CD-AI/Vintern-1B-v3_5", device=None):
    global _model, _tokenizer, _device
    if _model is None or _tokenizer is None:
        _device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval().to(_device)
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return _model, _tokenizer, _device


def _build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def _load_image(image, input_size=448):
    if image is None:
        raise ValueError("No image provided.")
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    transform = _build_transform(input_size)
    return transform(image).unsqueeze(0)


def process_image(image, user_request, model=None, tokenizer=None, device=None):
    """
    Xử lý ảnh và chạy qua mô hình Vintern.
    """
    if model is None or tokenizer is None:
        model, tokenizer, default_device = load_vlm_model()
        device = device or default_device
    elif device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Load and normalize image
    pixel_values = _load_image(image).to(device)

    generation_config = {
        "max_new_tokens": 256,
        "do_sample": False,
        "num_beams": 3,
        "repetition_penalty": 2.0
    }

    if not user_request or not user_request.strip():
        user_request = "Trích xuất toàn bộ thông tin trong ảnh và trả về dạng Markdown."

    question = f"<image>\n{user_request}"

    with torch.inference_mode():
        response, _ = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
            history=None,
            return_history=True
        )

    return response


def extract_patch(img, start, end, all_line_ys, show=False):
    y1 = all_line_ys[start]
    y2 = all_line_ys[end]
    patch = img[y1:y2, :]
    if show:
        pass  # local run: skip cv2_imshow (Colab-only), visualization disabled
    return patch


def read_patch(img, *, user_request=None, model=None, tokenizer=None, device=None, show=False):
    image = Image.fromarray(img)
    text = process_image(image, user_request, model, tokenizer, device)
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

            # OCR with Vintern
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
