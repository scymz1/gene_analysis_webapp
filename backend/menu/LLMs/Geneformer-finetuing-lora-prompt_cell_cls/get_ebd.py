import torch
from transformers import BertForTokenClassification,BertForSequenceClassification
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from embeds_loader import embeds
import numpy as np

pretrained_model_name = "geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/" #"geneformer-12L-30M/"



class geneformer(nn.Module):
    def __init__(self):
        super(geneformer, self).__init__()
        self.former = BertForSequenceClassification.from_pretrained(
        "geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/",
        num_labels=3,
        output_attentions=False,
        output_hidden_states=True
    )

    def forward(self,seq):
        output = self.former(seq)
        hidden_states = output['hidden_states'][6]
        x = torch.mean(hidden_states, dim=1)

        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = geneformer()
model = model.to(device)

directory_path = "/blue/qsong1/wang.qing/benchmark_dataset_API/Geneformer/samples/"
dt_list = os.listdir(directory_path)
for dt in dt_list:
    print('Processing: '+ dt)
    dataset = embeds(data_path=directory_path + dt)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False,num_workers=0)
    pbar = tqdm(dataloader)
    dataset_embeds = []
    with torch.no_grad():
        for batch in pbar:
            my_tensor = torch.tensor(batch).to(device)
            op = model(my_tensor.long())
            dataset_embeds.append(op.detach().cpu().numpy())

    dataset_embeds = np.vstack(dataset_embeds)
    np.save('/blue/qsong1/wang.qing/benchmark_dataset_API/Geneformer/embeds/'+dt[:-4]+'_embeds.npy', dataset_embeds)



