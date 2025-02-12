import csv
import os
import pickle
import numpy as np


with open('./gene_dict.pkl', 'rb') as file:
    dictionary = pickle.load(file)

directory_path = "/blue/qsong1/wang.qing/benchmark_dataset_API/lxndt_filter/"


def rebuilder(directory_path='/', is_sorted=True, seq_length=8192):
    csv.field_size_limit(500000000)
    files_list = os.listdir(directory_path)

    for filename in files_list:
        labels_cell = []
        samples = []
        csv_file_path = directory_path + filename
        print('processing ' + filename)
        header = True
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            pattern = []
            # sequence level
            for row in csv_reader:
                row_data = row[0].split('\t')
                if header:
                    for gene in row_data:
                        gene = gene.replace('"', '')
                        if gene in dictionary:
                            pattern.append(gene)
                        else:
                            pattern.append('none')
                    header = False
                    # print(pattern)
                else:
                    if len(pattern)!=len(row_data):
                        continue
                    seq_pattern_order_id_EXPscore = []
                    # token level
                    for i in range(len(row_data)):
                        if i==0:
                            pass
                        elif i==1:
                            if 'sensitive' in row_data[i]:
                                labels_cell.append(1)
                            elif 'resistant' in row_data[i]:
                                labels_cell.append(0)
                        else:
                            if pattern[i] == 'none':  # none token
                                pass
                            else:
                                seq_pattern_order_id_EXPscore.append((pattern[i], row_data[i]))
                    if is_sorted:
                        seq_pattern_order_id_EXPscore = sorted(seq_pattern_order_id_EXPscore, key=lambda x: x[1], reverse=True)

                    sample = [int(item[1]) for item in seq_pattern_order_id_EXPscore]

                    while len(sample)<=seq_length:
                        sample.append(0)
                    sample = sample[:seq_length]
                    samples.append(sample)


        np_samples = np.array(samples)
        np_labels = np.array(labels_cell)
        np.save('/blue/qsong1/wang.qing/benchmark_dataset_API/scBert/samples/'+filename[:-4]+'_samples.npy', np_samples)
        np.save('/blue/qsong1/wang.qing/benchmark_dataset_API/scBert/labels/'+filename[:-4]+'_labels.npy', np_labels)
        print(filename+' saved')


rebuilder(directory_path)

