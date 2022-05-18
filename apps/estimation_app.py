import dash_bootstrap_components as dbc
from flask import Response
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app
from app import server
import cv2
import numpy as np
from openvino_pipeline import OpenVINOPipeline, FaceDetectionModel, FacialLandmarkDetection, GazeEstimation, HeadPoseEstimation
from openvino_pipeline import visualize_result


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        width = self.video.get(3)
        height = self.video.get(4)
        if not success:
            image = np.zeros((height, width))
        return image

class PlaceholderCamera:
    @staticmethod
    def get_frame():
        return np.zeros((480, 640))

class PlaceholderPipeline:
    def __call__(self, img):
        return [[]] * 5

def gen(camera, pipeline):
    while True:
        frame = camera.get_frame()
        result = pipeline(frame)
        frame = visualize_result(frame, result)
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@server.route('/video_feed')
def video_feed():
    print("We are in video feed")
    face_detector = FaceDetectionModel("openvino_models/intel/face-detection-retail-0004/FP16")
    landmarks_model = FacialLandmarkDetection("openvino_models/intel/landmarks-regression-retail-0009/FP16")
    head_pose_model = HeadPoseEstimation("openvino_models/intel/head-pose-estimation-adas-0001/FP16")
    gaze_model = GazeEstimation("openvino_models/intel/gaze-estimation-adas-0002/FP16")
    print("We are in video feed and loaded models")
    return Response(gen(VideoCamera(), OpenVINOPipeline(face_detector, landmarks_model, head_pose_model, gaze_model)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@server.route("/placeholder_video_feed")
def placeholder_video_feed():
    print("We are in placeholder")
    return Response(gen(PlaceholderCamera(), PlaceholderPipeline()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.callback([
Output("rec-img", "src"), Output("rec-button", "children")
],
[Input("rec-button", "n_clicks")])
def start_record(n_clicks):
    print("Button clicked:", n_clicks)
    if n_clicks % 2 == 0:
        return "/placeholder_video_feed", "Start Video"
    else:
        return "/video_feed", "Stop Video"


layout = html.Div(
    [   dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Github Link", href='https://github.com/Lolik-Bolik/engagement_estimation_system')),
            dbc.NavItem(dbc.NavLink("Run Estimation", href='/apps/estimation_app')),
        ],
        brand="Engagement Estimation",
        brand_href="/",
        color="primary",
        dark=True,
    ),
        html.H2("Camera"),
        html.Div([
            html.Img(id="rec-img", src="/placeholder_video_feed"),
            html.Button('Start Video', id="rec-button", n_clicks=0, className="graphButtons")
        ],
        style={
            "text-align": "center"
        })



    ],
    style={
        "background-image": 'url(/assets/background_2.png)',
        "background-repeat": "no-repeat",
        "background-position": "center",
        "background-size": "cover",
        "position": "fixed",
	    "min-height": "100%",
	    "min-width": "100%",}
)