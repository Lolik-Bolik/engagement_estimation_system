import math
import cv2


def pad_img(img, pad_value, target_dims):
    h, w, _ = img.shape
    pads = []
    pads.append(int(math.floor((target_dims[0] - h) / 2.0)))
    pads.append(int(math.floor((target_dims[1] - w) / 2.0)))
    pads.append(int(target_dims[0] - h - pads[0]))
    pads.append(int(target_dims[1] - w - pads[1]))
    padded_img = cv2.copyMakeBorder(
        img, pads[0], pads[2], pads[1], pads[3], cv2.BORDER_CONSTANT, value=pad_value,
    )
    return padded_img, pads