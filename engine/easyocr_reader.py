import cv2
import numpy as np
import easyocr

_reader = None


def get_reader(langs=('vi',)):
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(list(langs))
    return _reader


def crop_by_keywords(cropped, top_word="THÔNG TIN HỌC SINH", bottom_words="A và C", c=0.1, debug=False):
    reader = get_reader()
    results = reader.readtext(cropped)
    top_word = top_word.lower()
    bottom_words = bottom_words.lower()
    top_box = None
    bottom_box = None

    for (bbox, text, conf) in results:
        t = text.lower()
        if top_word in t:
            top_box = bbox
        if bottom_words in t:
            bottom_box = bbox

    if top_box is None or bottom_box is None:
        raise ValueError(f"Could not find '{top_word}' or any of {bottom_words}")

    # compute diagonal crop coordinates
    x_top_left = min(p[0] for p in top_box)
    y_top_left = min(p[1] for p in top_box)
    x_bottom_right = max(p[0] for p in bottom_box)
    y_bottom_right = max(p[1] for p in bottom_box)

    # pad a bit
    word_height = max(p[1] for p in top_box) - min(p[1] for p in top_box)
    pad = int(word_height * c)

    x1 = max(0, int(x_top_left) - pad)
    y1 = max(0, int(y_top_left) - pad)
    x2 = min(cropped.shape[1], int(x_bottom_right) + pad * 8)
    y2 = min(cropped.shape[0], int(y_bottom_right) + pad * 8)

    cropped_section = cropped[y1:y2, x1:x2]

    if debug:
        pass  # local run: skip cv2_imshow (Colab-only), visualization disabled

    return cropped_section


def find_word_position(img, all_line_ys, word, debug=False):
    reader = get_reader()
    results = reader.readtext(img)

    word_y_center = None

    for (bbox, text, conf) in results:
        if word in text:
            y_coords = [p[1] for p in bbox]
            word_y_center = np.mean(y_coords)
            if debug:
                for (x, y) in bbox:
                    cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
                cv2.putText(img, word, (int(bbox[0][0]), int(bbox[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            break

    if word_y_center is None:
        raise ValueError(f"Word '{word}' not found in image")

    start = None
    end = None
    for i, y in enumerate(all_line_ys):
        if y < word_y_center:
            start = i
        elif y > word_y_center and end is None:
            end = i
            break

    if start is None:
        start = 0
    if end is None:
        end = len(all_line_ys) - 1

    if debug:
        pass  # local run: skip cv2_imshow (Colab-only), visualization disabled

    return start, end
