import os

import cv2
import numpy as np
from openvino.inference_engine import IECore
from .utils import pad_img


class FaceDetectionModel:
    def __init__(self, model_path, threshold=0.5):
        model_xml = os.path.join(model_path, "face-detection-retail-0004.xml")
        model_bin = os.path.join(model_path, "face-detection-retail-0004.bin")
        self.net = IECore().load_network(IECore().read_network(model=model_xml, weights=model_bin),
                                         device_name="CPU",
                                         num_requests=2)
        self.input_name = next(iter(self.net.input_info))
        self.output_name = sorted(self.net.outputs)[0]
        _, _, self.input_height, self.input_width = self.net.input_info[
            self.input_name
        ].input_data.shape
        self.threshold = threshold
        self.class_names = {
            0: "background",
            1: "face",
        }

    def preprocess(self, img):
        height, width, _ = img.shape
        if self.input_height / self.input_width < height / width:
            scale = self.input_height / height
        else:
            scale = self.input_width / width

        scaled_img = cv2.resize(
            img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
        )
        padded_img, pad = pad_img(
            scaled_img, (0, 0, 0), [self.input_height, self.input_width],
        )

        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = padded_img[np.newaxis].astype(np.float32)

        return [[padded_img], (scale, pad)]

    def postprocess(self, model_result):
        if not len(model_result[0]):
            return [None]
        result, input_info = model_result
        scale, pads = input_info
        h, w = self.input_height, self.input_width
        original_h = int((h - (pads[0] + pads[2])) / scale)
        original_w = int((w - (pads[1] + pads[3])) / scale)
        boxes = np.squeeze(result[0][self.output_name])
        image_predictions = []
        for box in boxes:
            if box[2] > self.threshold:  # confidence
                x1 = float(
                    np.clip((box[3] * w - pads[1]) / scale, 0, original_w),
                )
                y1 = float(
                    np.clip((box[4] * h - pads[0]) / scale, 0, original_h),
                )
                x2 = float(
                    np.clip((box[5] * w - pads[1]) / scale, 0, original_w),
                )
                y2 = float(
                    np.clip((box[6] * h - pads[0]) / scale, 0, original_h),
                )
                score = float(box[2])
                # class_name = self.class_names[int(box[1])]
                image_predictions.append((x1, y1, x2, y2, score))

        return np.array(image_predictions)

    def forward(self, data):
        data[0] = [
            self.net.infer(inputs={self.input_name: sample}) for sample in data[0]
        ]
        return data

    def process_sample(self, image):
        data = self.preprocess(image)
        output = self.forward(data)
        results = self.postprocess(output)
        return results
