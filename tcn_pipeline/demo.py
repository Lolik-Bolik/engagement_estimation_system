import os
from tqdm import tqdm
import torch
from torch import nn
import collections

import numpy as np

# from ResTCN import ResTCN
from ResTCN import ResTCN
import torchvision
import cv2
import time



def preprocess_frame(frame, transform):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = ((frame / 255) - means) / stds
    frame = torch.from_numpy(frame)
    # frame = frame.transpose((2, 0, 1))
    frame = frame.permute(2, 0, 1) / 255  # (H x W x C) to (C x H x W)
    frame = transform(frame)
    return frame

def run_net(network, input_tensor, device):
    with torch.no_grad():
        tic = time.time()
        inputs = input_tensor.to(device)
        print("Converting took ", time.time() - tic)
        outputs = network(inputs).squeeze()
        _, label = torch.max(outputs, dim=0)
        return label

def main():
    transform = torchvision.transforms.Compose([
        # torchvision.transforms.ToPILImage(),
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        torchvision.transforms.Resize([224, 224])
    ])
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device being used:", device, flush=True)

    cam = cv2.VideoCapture(0)
    model = ResTCN().to(device)
    model.load_state_dict(torch.load("model_epoch_17.pth"))
    model.eval()
    softmax = nn.Softmax()
    input_tensor = torch.empty((1, 16, 3, 224, 224), dtype=torch.float32)
    i = 0
    label = -1
    while True:
        i += 1
        if cv2.waitKey(3) == ord("q"):
            break
        ret, frame = cam.read()
        if not ret:
            break
        input_tensor[0, i] = preprocess_frame(frame, transform)
        if i == 15:
            i = 0
            tic = time.time()
            label = run_net(model, input_tensor, device)
            print("Prediction took ", time.time() - tic)
        cv2.putText(frame, f"Engagement Level: {label}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.imshow("out", frame)

if __name__ == "__main__":
    main()