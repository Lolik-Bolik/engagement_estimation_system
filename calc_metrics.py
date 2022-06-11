import os
from pytorch_pfn_extras import to
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import torch
from torch import nn, optim
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

import numpy as np

# from ResTCN import ResTCN
from ResTCN import ResTCN
from utils import get_dataloader

torch.manual_seed(0)

batch_size = 1

model_path = "saves/1/model_epoch_75.pth"
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
print("Device being used:", device, flush=True)


dataloader = get_dataloader(batch_size,
                            'test.csv',
                            os.path.join(os.getcwd(), 'data/DAiSEE/DataSet/TestFrames'),
                            'validation.csv',
                            os.path.join(os.getcwd(), 'data/DAiSEE/DataSet/ValidationFrames'))


dataset_sizes = {x: len(dataloader[x].dataset) for x in ['train', 'test']}
print(dataset_sizes, flush=True)


model = ResTCN().to(device)
model.load_state_dict(torch.load(model_path))

softmax = nn.Softmax()

y_trues = np.empty([0])
y_preds = np.empty([0])

model.eval()

for inputs, labels in tqdm(dataloader["train"], disable=False):
    inputs = inputs.to(device)
    labels = labels.long().squeeze().to(device)

    with torch.inference_mode():
        outputs = model(inputs).squeeze()

        preds = torch.max(softmax(outputs).unsqueeze(0), 1)[1]
        y_trues = np.append(y_trues, labels.data.cpu().numpy())
        y_preds = np.append(y_preds, preds.cpu())

print('\nconfusion matrix\n' + str(confusion_matrix(y_trues, y_preds)))
print('\naccuracy\t' + str(accuracy_score(y_trues, y_preds)))

