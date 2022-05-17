import argparse

import cv2
import numpy as np
from openvino_pipeline.face_detection import FaceDetectionModel
from openvino_pipeline.visualization import visualize_bbox


class VideoSource:
    def __init__(self, video_path=0):
        self.video = cv2.VideoCapture(video_path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        width = self.video.get(3)
        height = self.video.get(4)
        if not success:
            image = np.zeros((height, width))
        return image


class OpenVINOPipeline:
    def __init__(self, face_detector):
        self.face_detector = face_detector

    def __call__(self, img):
        return self.run_pipeline(img)

    def run_pipeline(self, img):
        bboxes = self.face_detector.process_sample(img)
        return bboxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cam', '--camera', action="store_true",
                        help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
    parser.add_argument('-vid', '--video', type=str,
                        help="Path to video file to be used for inference (conflicts with -cam)")
    parser.add_argument('-fd_path', '--face_detection_path', type=str,
                        help="Path to face detector weights")
    args = parser.parse_args()

    camera = not args.video

    if args.camera and args.video:
        raise ValueError("Incorrect command line parameters! \"-cam\" cannot be used with \"-vid\"!")
    elif args.camera is False and args.video is None:
        raise ValueError(
            "Missing inference source! Either use \"-cam\" to run on DepthAI camera or \"-vid <path>\" to run on video file")
    video_source = VideoSource(0 if camera else args.video)
    face_detector = FaceDetectionModel(args.face_detection_path)
    pipeline = OpenVINOPipeline(face_detector)
    while True:
        img = video_source.get_frame()
        bboxes = pipeline.run_pipeline(img)
        if bboxes is None:
            continue
        for bbox in bboxes:
            img = visualize_bbox(img, bbox, (10, 245, 10))
        cv2.imshow("", img)
        key = cv2.waitKey(5)
        if key == ord("q"):
            break
