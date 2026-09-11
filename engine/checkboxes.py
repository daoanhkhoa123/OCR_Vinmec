from functools import partial

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def detect_checkboxes(image, line_min_width=15, line_max_width=15):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray_scale, 150, 255, cv2.THRESH_BINARY)
    img_bin = ~img_bin

    kernal_h_min = np.ones((1, line_min_width), np.uint8)
    kernal_v_min = np.ones((line_min_width, 1), np.uint8)
    kernal_h_max = np.ones((1, line_max_width), np.uint8)
    kernal_v_max = np.ones((line_max_width, 1), np.uint8)

    img_bin_h_min = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h_min)
    img_bin_v_min = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v_min)
    img_bin_h_max = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernal_h_max)
    img_bin_v_max = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernal_v_max)

    img_bin_final = (img_bin_h_min & img_bin_h_max) | (img_bin_v_min & img_bin_v_max)
    img_bin_final = cv2.dilate(img_bin_final, np.ones((3, 3), np.uint8), iterations=1)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
    return stats, labels


def group_checkboxes_by_y(checkboxes, n_groups=13):
    if len(checkboxes) == 0:
        return [[]]
    y_centers = np.array([y + h / 2 for (x, y, w, h, label, conf) in checkboxes]).reshape(-1, 1)
    n_clusters = min(n_groups, len(checkboxes))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = kmeans.fit_predict(y_centers)
    groups = [[] for _ in range(n_clusters)]
    for i, lbl in enumerate(labels):
        groups[lbl].append(checkboxes[i])
    groups = sorted(groups, key=lambda g: np.mean([y + h / 2 for (_, y, _, h, _, _) in g]))
    for g in groups:
        g.sort(key=lambda box: box[0])
    return groups


def classify_checkboxes(image, stats, min_width=10, max_width=50, tick_threshold=0.1, n_groups=13):
    checkboxes = []
    for stat in stats[2:]:
        x, y, w, h, area = stat
        aspect_ratio = w / h
        if min_width <= w <= max_width and min_width <= h <= max_width and 0.8 <= aspect_ratio <= 1.2:
            roi = image[y:y + h, x:x + w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            binary_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)
            binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            tick_percentage = cv2.countNonZero(binary_roi) / (w * h)
            label = 'Ticked' if tick_percentage > tick_threshold else 'Unticked'
            checkboxes.append((x, y, w, h, label, tick_percentage))

    # group by Y center
    groups = group_checkboxes_by_y(checkboxes, n_groups)

    # flatten into dict with group index
    result = {}
    idx = 0
    for g_idx, g in enumerate(groups):
        for (x, y, w, h, label, conf) in g:
            result[idx] = {
                'position': (int(x), int(y), int(w), int(h)),
                'ticked': True if label == 'Ticked' else False,
                'tick_conf': float(conf),
                'group': int(g_idx + 1)
            }
            idx += 1

    return result


def plot_checkbox_groups(image, checkbox_dict):
    annotated = image.copy()
    for idx, data in checkbox_dict.items():
        x, y, w, h = data['position']
        group = data['group']
        label = data['ticked']
        color = (0, 255, 0) if label == 'Ticked' else (0, 0, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, f"G{group}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


BL1 = ["Nam", "Nữ"]
BL2 = ["Không", "Thuốc tra Atropine", "Kính gọng KSCT", "Ortho K", "Ánh sáng đỏ"]
BL3 = ["Mắt trầy xước", "Gọng cong vênh", "Sai tâm kính"]
BL4 = ["Có", "Không"]
BL5 = ["Đạt", "Không đạt"]
BL6 = ["Đạt", "Không đạt"]
BL7 = ["Đạt", "Không đạt"]
BL8 = ["Đạt", "Không đạt"]
BL9 = ["Hai mắt chính thị hoặc ít nguy cơ"]
BL10 = ["Nghi ngờ tật khúc xạ/bệnh lý mắt, cần khám để chẩn đoán xác định"]
BL11 = ["Có tật khúc xạ, kính đang đeo phù hợp"]
BL12 = ["Có tật khúc xạ, kính chưa tối ưu, cần kiểm tra lại kính đang đeo"]
BL13 = ["Cần đi khám chuyên khoa mắt (B và D)", "Ít nguy cơ (A và C)"]
DESC = ["Giới tính", "Đang điều trị kiểm soát cận thị", "KIỂM TRA KÍNH ĐANG ĐEO",
        "Đang đeo kính", "Đánh giá thị lực nhìn xa",
        "Thị lực nhìn gần (áp dụng với trẻ từ 4 đến 9 tuổi)", "Kết quả thị lực",
        "Kiểm tra sắc giác", "Hai mắt chính thị hoặc ít nguy cơ",
        "Nghi ngờ tật khúc xạ/bệnh lý mắt, cần khám để chẩn đoán xác định",
        "Có tật khúc xạ, kính đang đeo phù hợp",
        "Có tật khúc xạ, kính chưa tối ưu, cần kiểm tra lại kính đang đeo",
        "Kết luận khám"]


def bl2_only_no(checkbox_dict):
    res = []
    if checkbox_dict[0]["ticked"]:
        return res

    idx = 0
    for i, k in enumerate(checkbox_dict):
        if checkbox_dict[k]["group"] == 2:
            if checkbox_dict[k]["ticked"]:
                res.append(BL2[idx])
            idx += 1

    return res


def bl_only_one(checkbox_dict, group, field_names):
    min_thres = 1
    min_idx = 0
    idx = 0
    for i, k in enumerate(checkbox_dict):
        if checkbox_dict[k]["group"] == group:
            if checkbox_dict[k]["ticked"]:
                thres = checkbox_dict[k]["tick_conf"]
                if thres < min_thres:
                    min_thres = thres
                    min_idx = idx
            idx += 1

    return field_names[min_idx]


def bl_get_all(checkbox_dict, group, field_names):
    res = []
    idx = 0
    for i, k in enumerate(checkbox_dict):
        if checkbox_dict[k]["group"] == group:
            if checkbox_dict[k]["ticked"]:
                res.append(field_names[idx])
            idx += 1

    return res


BL_f = [partial(bl_only_one, group=1, field_names=BL1),
        bl2_only_no,
        partial(bl_get_all, group=3, field_names=BL3),
        partial(bl_only_one, group=4, field_names=BL4),
        *[partial(bl_only_one, group=g, field_names=BL5)
          for g in range(5, 8 + 1)],
        partial(bl_get_all, group=9, field_names=BL9),
        partial(bl_get_all, group=10, field_names=BL10),
        partial(bl_get_all, group=11, field_names=BL11),
        partial(bl_get_all, group=12, field_names=BL12),
        partial(bl_only_one, group=13, field_names=BL13)
        ]


def summarize_checkbox(checkbox_dict):
    d = {}
    for i, f in enumerate(BL_f):
        res = f(checkbox_dict)
        if not res:
            res = None

        if isinstance(res, list):
            res = res[0]

        d[DESC[i]] = res

    return d
