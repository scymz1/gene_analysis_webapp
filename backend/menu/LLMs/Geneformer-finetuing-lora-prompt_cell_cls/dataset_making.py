import csv
import os
import pickle
import numpy as np

with open('my_dict.pkl', 'rb') as file:
    dictionary = pickle.load(file)

directory_path = "/blue/qsong1/wang.qing/benchmark_dataset_API/lxndt_filter/"


def dataset_generator(directory_path='/', is_sorted=True, seq_length=2048):
    csv.field_size_limit(500000000)
    files_list = os.listdir(directory_path)

    for filename in files_list:
        csv_file_path = directory_path + filename
        print(csv_file_path, 'processing...')
        headr = True
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            pattern = []
            labels = []
            samples= []
            # sequence level
            for row in csv_reader:
                row_data = row[0].split('\t')
                if headr:
                    for gene in row_data:
                        gene = gene.replace('"', '')
                        if gene in dictionary:
                            pattern.append(dictionary[gene])
                        else:
                            pattern.append(-99999)
                    headr = False
                    # print(pattern)
                else:
                    assert len(pattern)==len(row_data)
                    seq_pattern_order_id_EXPscore = []
                    # token level
                    for i in range(len(row_data)):
                        if i==0:
                            pass
                        elif i==1:
                            if 'sensitive' in row_data[i]:
                                labels.append(1)
                            elif 'resistant' in row_data[i]:
                                labels.append(0)
                        else:
                            if row_data[i]=='0':
                                pass
                            else:
                                if pattern[i]==-99999: # none token
                                    pass
                                else:
                                    seq_pattern_order_id_EXPscore.append((pattern[i],row_data[i]))

                    if is_sorted:
                        seq_pattern_order_id_EXPscore = sorted(seq_pattern_order_id_EXPscore, key=lambda x: x[1], reverse=True)
                    sample = [item[0] for item in seq_pattern_order_id_EXPscore]

                    while len(sample)<=seq_length:
                        sample.append(0)
                    sample = sample[:seq_length]
                    samples.append(sample)


        # datset save path
        file_path_samples = '/blue/qsong1/wang.qing/benchmark_dataset_API/Geneformer/samples' + '/' + filename[:-4] + '_samples.npy'
        file_path_labels = '/blue/qsong1/wang.qing/benchmark_dataset_API/Geneformer/labels' + '/' + filename[:-4] + '_labels.npy'

        np.save(file_path_samples,np.array(samples))
        np.save(file_path_labels,np.array(labels))
        print('saved ' + filename)


dataset_generator(directory_path)




