import cv2

LINE_THINKNESS = 3
NORM_HEIGHT = 591


def visualize_bbox(img, bbox, bbox_color):
    img_h, img_w, _ = img.shape
    x1, y1, x2, y2 = bbox[:4]
    scale = min([img_w, img_h]) / NORM_HEIGHT
    thickness = int(scale * LINE_THINKNESS)
    img = cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        bbox_color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return img