import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import os


class HeadPoseEstimation:
    '''
    Class for the Head Pose Estimation Model.
    '''

    def __init__(self, model_path):
        model_xml = os.path.join(model_path, "head-pose-estimation-adas-0001.xml")
        model_bin = os.path.join(model_path, "head-pose-estimation-adas-0001.bin")
        self.core = IECore()
        self.head_pose = self.core.read_network(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(self.head_pose.input_info))
        self.out_blob = next(iter(self.head_pose.outputs))
        self.load_model()

    def load_model(self):
        self.exec_net = self.core.load_network(network=self.head_pose, device_name="CPU")
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
        supported_layer_map = self.core.query_network(network=self.head_pose, device_name="CPU")
        supported_layers = supported_layer_map.keys()
        unsupported_layer_exists = False
        network_layers = self.head_pose.layers.keys()
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
            print("[INFO][Head Pose Detection Model] All layers are suported")

    def preprocess_input(self, image):

        n, c, h, w = self.head_pose.input_info[self.input_blob].input_data.shape

        image = cv2.resize(image, (w, h))
        image = image.transpose(2, 0, 1)
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs):

        pitch = outputs['angle_p_fc'][0][0]
        roll = outputs['angle_r_fc'][0][0]
        yaw = outputs['angle_y_fc'][0][0]

        return pitch, roll, yaw

    def process_sample(self, image):
        data = self.preprocess_input(image)
        output = self.sync_inference(data)
        results = self.preprocess_output(output)
        return results
