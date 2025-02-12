
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2Model

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer_file = "lixiangchun/transcriptome-gpt-1024-8-16-64"
checkpoint = "lixiangchun/transcriptome-gpt-1024-8-16-64" ## Pretrained model


class LineDataset(Dataset):
    def __init__(self, f_path):
        self.lines = np.load(f_path, allow_pickle=True)['array']
        pass
    def __getitem__(self, i):
        return torch.tensor(self.lines[i])
    def __len__(self):
        return len(self.lines)

tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_file)
model = GPT2LMHeadModel.from_pretrained(checkpoint,output_hidden_states = True).transformer
model = model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")


directory_path = "/blue/qsong1/wang.qing/benchmark_dataset_API/tGPT/samples/"
dt_list = os.listdir(directory_path)
for dt in dt_list:
    ds = LineDataset(directory_path+dt)
    dl = DataLoader(ds, batch_size=64, shuffle=False)

    Xs = []
    for bs in tqdm(dl, total=len(dl)):
        bs = bs.to(device)
        with torch.no_grad():
            x = model(bs)

        xx = x.last_hidden_state
        xx = torch.mean(xx, dim=1).tolist()

        Xs.extend(xx)
        pass

    features = np.stack(Xs)
    np.save('/blue/qsong1/wang.qing/benchmark_dataset_API/tGPT/embeds/' + dt[:-12] + '_embeds.npy', features)
    print('saved ' + dt)