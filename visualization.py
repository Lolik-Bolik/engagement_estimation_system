import cv2
import numpy as np
from math import cos, sin

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


def visualize_label(img, label):
    cv2.putText(img, f"Engagement Level: {label}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                cv2.LINE_AA)
    return img


def draw_3d_axis(image, head_pose, origin, size=50):
    # From https://github.com/openvinotoolkit/open_model_zoo/blob/b1ff98b64a6222cf6b5f3838dc0271422250de95/demos/gaze_estimation_demo/cpp/src/results_marker.cpp#L50
    origin_x, origin_y = origin
    yaw, pitch, roll = np.array(head_pose)*np.pi / 180

    sinY = sin(yaw)
    sinP = sin(pitch)
    sinR = sin(roll)

    cosY = cos(yaw)
    cosP = cos(pitch)
    cosR = cos(roll)
    # X axis (red)
    x1 = origin_x + size * (cosR * cosY + sinY * sinP * sinR)
    y1 = origin_y + size * cosP * sinR
    cv2.line(image, (origin_x, origin_y), (int(x1), int(y1)), (0, 0, 255), 3)

    # Y axis (green)
    x2 = origin_x + size * (cosR * sinY * sinP + cosY * sinR)
    y2 = origin_y - size * cosP * cosR
    cv2.line(image, (origin_x, origin_y), (int(x2), int(y2)), (0, 255, 0), 3)

    # Z axis (blue)
    x3 = origin_x + size * (sinY * cosP)
    y3 = origin_y + size * sinP
    cv2.line(image, (origin_x, origin_y), (int(x3), int(y3)), (255, 0, 0), 2)

    return image


def draw_gaze(img, left_eye_coord, right_eye_coord, gaze):
    re_x = (int(right_eye_coord[0] + right_eye_coord[2])) // 2
    re_y = (int(right_eye_coord[1] + right_eye_coord[3])) // 2
    le_x = (int(left_eye_coord[0] + left_eye_coord[2])) // 2
    le_y = (int(left_eye_coord[1] + left_eye_coord[3])) // 2

    x, y = (gaze * 100).astype(int)[:2]
    img = cv2.arrowedLine(img, (le_x, le_y), (le_x + x, le_y - y), (255, 0, 255), 3)
    img = cv2.arrowedLine(img, (re_x, re_y), (re_x + x, re_y - y), (255, 0, 255), 3)

    return img

