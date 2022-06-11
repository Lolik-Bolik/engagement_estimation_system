import os
from tqdm import tqdm
import pandas as pd 

phases = ['Train', 'Test', 'Validation']

def img_to_label(df: pd.DataFrame):
    img_lbl_dict = {}
    for line in df.iterrows():
        name = line[1]["ClipID"].split(".")[0]
        lbl = line[1]["Engagement"]
        img_lbl_dict[name] = lbl

    return img_lbl_dict

working_dir = os.getcwd()

for phase in phases:

    path = os.path.join('data/DAiSEE/DataSet', phase+"Frames")
    subjects = os.listdir(path)
    labels = pd.read_csv(f'data/DAiSEE/Labels/AllLabels.csv')

    data_collection = ["path,label\n"]

    img_to_lbl_dict = img_to_label(labels)

    for subject in subjects:
        videos = os.listdir(os.path.join(path, subject))
        for video in tqdm(videos):
            local_path = os.path.join(working_dir, path, subject, video)
            lbl = img_to_lbl_dict.get(video, None)
            if lbl is None:
                continue
            data_collection.append(f"{local_path},{lbl}\n")
    
    with open(os.path.join(working_dir, phase.lower()+".csv"), "w") as f:
        f.writelines(data_collection)
