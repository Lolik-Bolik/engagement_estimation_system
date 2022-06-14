import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
from torch import nn, optim
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

import numpy as np

# from ResTCN import ResTCN
from ResTCN import ResTCN
from utils import generate_dataloader

torch.manual_seed(0)
batch_size = 16
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
print("Device being used:", device, flush=True)


dataloader = generate_dataloader(batch_size,
                            'test.csv',
                            '/home/pavel/datasets/maga/NIS/DAiSEE_test/TestFrames')
dataset_sizes = {"test": len(dataloader.dataset)}
print(dataset_sizes, flush=True)


model = ResTCN().to(device)
model.load_state_dict(torch.load("model_epoch_17.pth"))
softmax = nn.Softmax()
criterion = nn.CrossEntropyLoss().to(device)

running_loss = .0
y_trues = np.empty([0])
y_preds = np.empty([0])
model.eval()

for inputs, labels in tqdm(dataloader, disable=False):
    inputs = inputs.to(device)
    labels = labels.long().squeeze().to(device)

    with torch.no_grad():
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)

    running_loss += loss.item() * inputs.size(0)
    preds = torch.max(softmax(outputs), 1)[1]
    y_trues = np.append(y_trues, labels.data.cpu().numpy())
    y_preds = np.append(y_preds, preds.cpu())

# if phase == 'train':
#     scheduler.step()

epoch_loss = running_loss / dataset_sizes["test"]

print("[{}] Loss: {}".format(
    "test", epoch_loss), flush=True)
print('\nconfusion matrix\n' + str(confusion_matrix(y_trues, y_preds)))
print('\naccuracy\t' + str(accuracy_score(y_trues, y_preds)))

exp_id = "exp_0"
save_path = os.path.join(os.getcwd(), "saves", str(exp_id))
if not os.path.exists(save_path):
    os.makedirs(save_path)
