import csv
import os
import pickle
import numpy as np
import pandas as pd

directory_path = "/blue/qsong1/wang.qing/benchmark_dataset_API/lxndt_filter/"


def dataset_generator(directory_path='/', is_sorted=True, seq_length=8192):
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

            for row in csv_reader:
                row_data = row[0].split('\t')
                if header:
                    for gene in row_data:
                        gene = gene.replace('"', '')
                        pattern.append(gene)
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
                             seq_pattern_order_id_EXPscore.append((pattern[i],row_data[i]))
                    if is_sorted:
                        seq_pattern_order_id_EXPscore = sorted(seq_pattern_order_id_EXPscore, key=lambda x: x[1], reverse=True)
                    sample = [int(item[1]) for item in seq_pattern_order_id_EXPscore]
                    while len(sample)<=seq_length:
                        sample.append(0)
                    sample = sample[:seq_length]
                    samples.append(sample)



        np_samples = np.array(samples)
        np_labels = np.array(labels_cell)


        np.savez('/blue/qsong1/wang.qing/benchmark_dataset_API/UCE/samples/'+filename[:-4]+'_samples.npz', array=np_samples)
        np.savez('/blue/qsong1/wang.qing/benchmark_dataset_API/UCE/labels/'+filename[:-4]+'_labels.npz', array=np_labels)
        print('saved ' + filename)





dataset_generator(directory_path)


def shape(directory_path_samples,directory_path_labels):
    print('samples shape')
    files_list = os.listdir(directory_path_samples)
    dict_samples_shape={}
    for filename in files_list:
        loaded_compressed = np.load(directory_path_samples+filename)
        loaded_compressed_array = loaded_compressed['array']
        dict_samples_shape[filename] = loaded_compressed_array.shape
        print(filename, loaded_compressed_array.shape)

    with open('./dict_samples_shape.pkl', 'wb') as f:
        pickle.dump(dict_samples_shape, f)

    print('*'*200)
    print('labels shape')
    
    files_list = os.listdir(directory_path_labels)
    dict_labels_shape={}
    for filename in files_list:
        loaded_compressed = np.load(directory_path_labels+filename)
        loaded_compressed_array = loaded_compressed['array']
        dict_labels_shape[filename] = loaded_compressed_array.shape
        print(filename, loaded_compressed_array.shape)

    with open('./dict_labels_shape.pkl', 'wb') as f:
        pickle.dump(dict_labels_shape, f)

directory_path_samples = "/blue/qsong1/wang.qing/benchmark_dataset_API/UCE/samples/"
directory_path_labels = "/blue/qsong1/wang.qing/benchmark_dataset_API/UCE/labels/"

shape(directory_path_samples,directory_path_labels)