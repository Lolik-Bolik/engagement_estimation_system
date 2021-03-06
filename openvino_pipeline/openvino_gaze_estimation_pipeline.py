import argparse

import cv2
import numpy as np
from openvino_pipeline.face_detection import FaceDetectionModel
from openvino_pipeline.facial_landmarks import FacialLandmarkDetection
from openvino_pipeline.visualization import visualize_bbox, draw_3d_axis, draw_gaze
from openvino_pipeline.head_pose import HeadPoseEstimation
from openvino_pipeline.gaze_estimation import GazeEstimation


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
    def __init__(self, face_detector, landmarks_model, head_pose_model, gaze_model):
        self.face_detector = face_detector
        self.landmarks_model = landmarks_model
        self.head_pose_model = head_pose_model
        self.gaze_model = gaze_model

    def __call__(self, img):
        return self.run_pipeline(img)

    @staticmethod
    def clip_bbox(img, bbox):
        max_w, max_h, _ = img.shape
        bbox[0::2] = np.clip(bbox[0::2], 0, max_w)
        bbox[1::2] = np.clip(bbox[1::2], 0, max_h)
        return bbox

    def get_face_crops(self, img, bboxes):
        if not bboxes.size:
            return None

        faces = []
        bboxes = bboxes.astype(np.int32)
        for bbox in bboxes:
            bbox = self.clip_bbox(img, bbox)
            x1, y1, x2, y2 = bbox[:4]
            crop = img[y1:y2, x1:x2]
            faces.append(crop)

        faces = faces[0]
        return faces

    def get_eye_crops(self, face, left_eye_coord, right_eye_coord):
        left_eye_coord = self.clip_bbox(face, left_eye_coord)
        x1, y1, x2, y2 = left_eye_coord
        left_eye = face[y1:y2, x1:x2]

        right_eye_coord = self.clip_bbox(face, right_eye_coord)
        x1, y1, x2, y2 = right_eye_coord
        right_eye = face[y1:y2, x1:x2]

        if len(left_eye) == 0 and len(right_eye) != 0:
            left_eye = right_eye.copy()
        elif len(right_eye) == 0 and len(left_eye) != 0:
            right_eye = left_eye.copy()
        elif len(left_eye) == 0 and len(right_eye) == 0:
            return None, None

        return left_eye, right_eye

    def run_pipeline(self, img):
        bboxes = self.face_detector.process_sample(img)
        faces = self.get_face_crops(img, bboxes)
        if faces is not None and faces.size > 0:
            head_pose = self.head_pose_model.process_sample(faces)
            landmarks, left_eye_coord, right_eye_coord = self.landmarks_model.process_sample(faces)
            left_eye, right_eye = self.get_eye_crops(faces, left_eye_coord, right_eye_coord)
            if left_eye is not None and right_eye is not None:
                gaze = self.gaze_model.process_sample(head_pose, left_eye, right_eye)
            else:
                gaze = None
            return bboxes, left_eye_coord, right_eye_coord, head_pose, gaze
        else:
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cam', '--camera', action="store_true",
                        help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
    parser.add_argument('-vid', '--video', type=str,
                        help="Path to video file to be used for inference (conflicts with -cam)")
    parser.add_argument('-fd_path', '--face_detection_path', type=str,
                        help="Path to face detector weights")
    parser.add_argument('-lm_path', '--landmarks_model_path', type=str,
                        help="Path to facial landmarks model weights")
    parser.add_argument('-hp_path', '--head_pose_path', type=str,
                        help="Path to head pose estimation model")
    parser.add_argument('-g_path', '--gaze_path', type=str,
                        help="Path to gaze estimation model")
    args = parser.parse_args()

    camera = not args.video

    if args.camera and args.video:
        raise ValueError("Incorrect command line parameters! \"-cam\" cannot be used with \"-vid\"!")
    elif args.camera is False and args.video is None:
        raise ValueError(
            "Missing inference source! Either use \"-cam\" to run on DepthAI camera or \"-vid <path>\" to run on video file")
    video_source = VideoSource(0 if camera else args.video)
    face_detector = FaceDetectionModel(args.face_detection_path)
    landmarks_model = FacialLandmarkDetection(args.landmarks_model_path)
    head_pose_model = HeadPoseEstimation(args.head_pose_path)
    gaze_model = GazeEstimation(args.gaze_path)

    pipeline = OpenVINOPipeline(face_detector, landmarks_model, head_pose_model, gaze_model)
    while True:
        img = video_source.get_frame()
        result = pipeline.run_pipeline(img)

        if result is not None:
            bboxes, le, re, head_pose, gaze = result

            # TODO: landmark model with batch
            # assert len(bboxes) == 1

            for box in bboxes:
                hp_origin = (60, 60)
                box_1 = [le[0] + box[0], le[1] + box[1], le[2] + box[0], le[3] + box[1]]
                box_2 = [re[0] + box[0], re[1] + box[1], re[2] + box[0], re[3] + box[1]]

                img = visualize_bbox(img, box, (10, 245, 10))  # draw face box
                img = visualize_bbox(img, box_1, (255, 0, 0))  # draw eye boxes
                img = visualize_bbox(img, box_2, (255, 0, 0))

                img = draw_3d_axis(img, head_pose, hp_origin)  # draw head pose
                if gaze is not None:
                    img = draw_gaze(img, box_1, box_2, gaze)

        cv2.imshow("", img)
        key = cv2.waitKey(5)
        if key == ord("q"):
            break
