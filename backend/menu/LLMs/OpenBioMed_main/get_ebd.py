import os.path as osp
import sys
path = osp.dirname(osp.abspath(''))
sys.path.append(path)
sys.path.append(osp.join(path, "open_biomed"))



from open_biomed.datasets.ctc_dataset import Zheng68k
import json
from torch.utils.data import DataLoader
import os
from open_biomed.models.task_model.ctc_model import CTCModel
import torch
import pickle
config = json.load(open("./configs/ctc/cellLM.json", "r"))
model = CTCModel(config["network"], 2)


from tqdm import tqdm

dt_path="/blue/qsong1/wang.qing/benchmark_dataset_API/lxndt_filter/"


files_list = os.listdir(dt_path)

for filename in files_list:
    print(filename)
    dataset = Zheng68k(path=dt_path+filename,config=config["data"], seed=2023)
    loader = DataLoader(dataset, batch_size=48, shuffle=False)
    print('finished loading')

    with torch.no_grad():
        model.cuda()
        model.eval()
        all_preds, all_y = [], []
        for cell, label in tqdm(loader):
            cell = cell.cuda()
            embed = model(cell)

            all_preds.append(embed)
            all_y.append(label)

        all_preds = torch.cat(all_preds, dim=0).cpu()
        all_y = torch.cat(all_y, dim=0).cpu()
    data_dict={}
    data_dict["embeds"]=all_preds
    data_dict["labels"]=all_y
    with open('/blue/qsong1/wang.qing/benchmark_dataset_API/cellLM/samples/'+filename[:-4]+'_samples.pkl', 'wb') as pklf:
        pickle.dump(data_dict, pklf)