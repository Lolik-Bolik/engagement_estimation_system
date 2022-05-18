import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import os


class FacialLandmarkDetection:
    '''
    Class for the Facial Landmark Detection Model.
    '''

    def __init__(self, model_path, device='CPU', extensions=None):
        model_xml = os.path.join(model_path, "landmarks-regression-retail-0009.xml")
        model_bin = os.path.join(model_path, "landmarks-regression-retail-0009.bin")
        self.core = IECore()
        self.net = self.core.read_network(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(self.net.input_info))
        self.out_blob = next(iter(self.net.outputs))
        self.load_model()

    def load_model(self):

        self.exec_net = self.core.load_network(network=self.net, device_name="CPU")

        return self.exec_net

    def sync_inference(self, image):
        return self.exec_net.infer({self.input_blob: image})

    def async_inference(self, image, request_id=0):
        # create async network
        input_blob = next(iter(self.exec_net.inputs))
        async_net = self.exec_net.start_async(request_id, inputs={input_blob: image})

        # perform async inference
        output_blob = next(iter(async_net.outputs))
        status = async_net.requests[request_id].wait(-1)
        if status == 0:
            result = async_net.requests[request_id].outputs[output_blob]
        return result

    def check_model(self):

        supported_layer_map = self.core.query_network(network=self.net, device_name="CPU")
        supported_layers = supported_layer_map.keys()
        unsupported_layer_exists = False
        network_layers = self.net.layers.keys()
        for layer in network_layers:
            if layer in supported_layers:
                pass
            else:
                print("[INFO] {} Still Unsupported".format(layer))
                unsupported_layer_exists = True

        if unsupported_layer_exists:
            print("Exiting the program.")
            exit(1)
        else:
            print("[INFO][Facial Landmark Detection Model] All layers are suported")

    def preprocess_input(self, image):

        n, c, h, w = self.exec_net.input_info[self.input_blob].input_data.shape

        image = cv2.resize(image, (w, h))
        image = image.transpose(2, 0, 1)
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, image, outputs):

        width = image.shape[1]
        height = image.shape[0]

        # shape (1x70)
        facial_landmark = outputs[self.out_blob][0]

        # convert from [x0,y0,x1,y1,...,x34,y34] to [(x0,y0),(x1,y1),...,(x34,y34)] and scale to input size
        j = 0
        landmark_points = []
        for i in range(int(len(facial_landmark) / 2)):
            point = (int(facial_landmark[j] * width), int(facial_landmark[j + 1] * height))
            landmark_points.append(point)
            j += 2

        eye_dist = abs(landmark_points[1][0] - landmark_points[0][0]) * 0.33

        left_eye_coord = np.array([
            landmark_points[0][0] - eye_dist, landmark_points[0][1] - eye_dist / 2,
            landmark_points[0][0] + eye_dist, landmark_points[0][1] + eye_dist / 2
        ])
        right_eye_coord = np.array([
            landmark_points[1][0] - eye_dist, landmark_points[1][1] - eye_dist / 2,
            landmark_points[1][0] + eye_dist, landmark_points[1][1] + eye_dist / 2
        ])

        return landmark_points, left_eye_coord.astype(np.int32), right_eye_coord.astype(np.int32)

    def process_sample(self, image):
        data = self.preprocess_input(image)
        output = self.sync_inference(data)
        results = self.preprocess_output(image, output)
        return results