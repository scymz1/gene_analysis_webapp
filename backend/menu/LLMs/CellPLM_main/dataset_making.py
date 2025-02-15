import warnings
warnings.filterwarnings("ignore")


import numpy as np

from CellPLM.utils import set_seed
import os
import csv
import json
import torch
import pickle
import mygene

print(torch.version.cuda) 
with open('/blue/qsong1/wang.qing/benchmark_scLLM_API/CellPLM-main/CellPLM-main/ckpt/20230926_85M.config.json', 'r') as file:
    data = json.load(file)

model_gene_set = set(data['gene_list'])
def symbol_to_ensembl(gene_list):
    mg = mygene.MyGeneInfo()
    return mg.querymany(gene_list, scopes='symbol', fields='ensembl.gene', as_dataframe=True,
                 species='human').reset_index().drop_duplicates(subset='query')['ensembl.gene'].fillna('0').tolist()



set_seed(42)

csv.field_size_limit(500000000)
directory_path = '/blue/qsong1/wang.qing/benchmark_dataset_API/lxndt_filter/'
files_list = os.listdir(directory_path)


for filename in files_list:
    labels = []
    samples = []
    pattern = []
    dt = {}
    csv_file_path = directory_path + filename
    print('processing '+csv_file_path)
    head = True
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        # sequence level
        for row in csv_reader:
            row_data = row[0].split('\t')
            if head:
                for gene in row_data:
                    gene = gene.replace('"', '')
                    pattern.append(gene)
                pattern_ensembl = symbol_to_ensembl(pattern[2:])
                head = False
                # print(pattern)
            else:
                if len(pattern) != len(row_data):
                    continue
                seq_pattern_order_id_EXPscore = []
                # token level
                for i in range(len(row_data)):
                    if i == 0:
                        pass
                    elif i == 1:
                        if 'sensitive' in row_data[i]:
                            labels.append(1)
                        elif 'resistant' in row_data[i]:
                            labels.append(0)
                    else:
                        seq_pattern_order_id_EXPscore.append(row_data[i])
                seq = []
                gene_list = []
                for i in range(len(pattern_ensembl)):
                    if pattern_ensembl[i] in model_gene_set:
                        seq.append(int(seq_pattern_order_id_EXPscore[i]))
                        gene_list.append(pattern_ensembl[i])

                samples.append(torch.tensor(seq))

    adata = {}

    adata['x_seq'] = torch.stack(samples)
    adata['gene_list'] = gene_list
    with open('/blue/qsong1/wang.qing/benchmark_dataset_API/cellPLM/samples/' + filename[:-4] + '_samples.pkl', "wb") as f:
        pickle.dump(adata, f)

    np.save('/blue/qsong1/wang.qing/benchmark_dataset_API/cellPLM/labels/' + filename[:-4] + '_labels.npy', labels)
    print('saved '+ csv_file_path)