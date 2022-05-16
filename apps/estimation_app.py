import dash_bootstrap_components as dbc
from flask import Response
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app
from app import server
import cv2
import numpy as np

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if success:
            ret, jpeg = cv2.imencode('.jpg', image)
        else:
            ret, jpeg = cv2.imencode(".jpg", np.zeros((480, 640)))
        return jpeg.tobytes()

class PlaceholderCamera:
    @staticmethod
    def get_frame():
        ret, jpeg = cv2.imencode(".jpg", np.zeros((480, 640)))
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@server.route("/placeholder_video_feed")
def placeholder_video_feed():
    return Response(gen(PlaceholderCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.callback([
Output("rec-img", "src"), Output("rec-button", "children")
],
[Input("rec-button", "n_clicks")])
def start_record(n_clicks):
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
        html.Img(id="rec-img", src="/placeholder_video_feed"),
        html.Button('Start Video', id="rec-button",n_clicks=0, className = "graphButtons")


    ],
    style={
        "background-image": 'url(/assets/background_2.jpg)',
        "background-repeat": "no-repeat",
        "background-position": "center",
        "background-size": "cover",
        "position": "fixed",
	    "min-height": "100%",
	    "min-width": "100%",}
)