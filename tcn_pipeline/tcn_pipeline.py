import torch.cuda
import torch
import torchvision
import cv2


class TCNPipeline:
    def __init__(self, model, device):
        print("Init called")
        self.model = model
        self.device = device
        self.transform = torchvision.transforms.Compose([
        # torchvision.transforms.ToPILImage(),
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        torchvision.transforms.Resize([224, 224])
    ])
        self.model.to(self.device)
        print("Weights Loading started")
        self.model.load_state_dict(torch.load("/home/pavel/Labs/Mag_1_course_2nd_semester/engagement_estimation_system/tcn_pipeline/model_epoch_17.pth"))
        print("Weight Loading Finished")
        self.model.eval()
        self.input_tensor = torch.empty((1, 16, 3, 224, 224), dtype=torch.float32)
        self.running_idx = 0
        self.predicted_label = -1
        print("Init finished")

    def __call__(self, img):
        return self.run_pipeline(img)

    def preprocess_frame(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # frame = ((frame / 255) - means) / stds
        img = torch.from_numpy(img)
        # frame = frame.transpose((2, 0, 1))
        img = img.permute(2, 0, 1) / 255  # (H x W x C) to (C x H x W)
        img = self.transform(img)
        return img

    def run_net(self):
        with torch.no_grad():
            inputs = self.input_tensor.to(self.device)
            outputs = self.model(inputs).squeeze()
            _, label = torch.max(outputs, dim=0)
            self.predicted_label = label

    def run_pipeline(self, img):
        self.input_tensor[0, self.running_idx] = self.preprocess_frame(img)
        self.running_idx += 1
        if self.running_idx == 15:
            self.running_idx = 0
            self.run_net()
        return self.predicted_label
