import csv
import os
import pickle
import numpy as np
import pandas as pd

def dataset_generator(directory_path, output_dir, is_sorted=True, seq_length=8192):
    """
    Generate datasets from CSV files.
    
    Args:
        directory_path (str): Path to input directory containing CSV files
        output_dir (str): Path to output directory for saving processed files
        is_sorted (bool): Whether to sort the data
        seq_length (int): Length of sequences to generate
    """
    # Create samples and labels subdirectories
    samples_dir = os.path.join(output_dir, 'samples')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    csv.field_size_limit(500000000)
    files_list = os.listdir(directory_path)

    for filename in files_list:
        if not filename.endswith('.csv'):
            continue
            
        labels_cell = []
        samples = []
        csv_file_path = os.path.join(directory_path, filename)

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
                    sample = [int(float(item[1])) for item in seq_pattern_order_id_EXPscore]
                    while len(sample) < seq_length:
                        sample.append(0)
                    sample = sample[:seq_length]
                    samples.append(sample)

        np_samples = np.array(samples)
        np_labels = np.array(labels_cell)

        samples_path = os.path.join(samples_dir, f'{filename[:-4]}_samples.npz')
        labels_path = os.path.join(labels_dir, f'{filename[:-4]}_labels.npz')
        
        np.savez(samples_path, array=np_samples)
        np.savez(labels_path, array=np_labels)
        print('saved ' + filename)

    return {
        'samples_dir': samples_dir,
        'labels_dir': labels_dir
    }

def shape(directory_path_samples, directory_path_labels, output_dir):
    """
    Generate shape dictionaries for samples and labels.
    
    Args:
        directory_path_samples (str): Path to samples directory
        directory_path_labels (str): Path to labels directory
        output_dir (str): Path to output directory for saving shape dictionaries
    """
    print('samples shape')
    files_list = os.listdir(directory_path_samples)
    dict_samples_shape = {}
    for filename in files_list:
        loaded_compressed = np.load(os.path.join(directory_path_samples, filename))
        loaded_compressed_array = loaded_compressed['array']
        dict_samples_shape[filename] = loaded_compressed_array.shape
        print(filename, loaded_compressed_array.shape)

    samples_shape_path = os.path.join(output_dir, 'dict_samples_shape.pkl')
    with open(samples_shape_path, 'wb') as f:
        pickle.dump(dict_samples_shape, f)

    print('*'*200)
    print('labels shape')
    
    files_list = os.listdir(directory_path_labels)
    dict_labels_shape = {}
    for filename in files_list:
        loaded_compressed = np.load(os.path.join(directory_path_labels, filename))
        loaded_compressed_array = loaded_compressed['array']
        dict_labels_shape[filename] = loaded_compressed_array.shape
        print(filename, loaded_compressed_array.shape)

    labels_shape_path = os.path.join(output_dir, 'dict_labels_shape.pkl')
    with open(labels_shape_path, 'wb') as f:
        pickle.dump(dict_labels_shape, f)
        
    return {
        'samples_shape_path': samples_shape_path,
        'labels_shape_path': labels_shape_path
    }

# directory_path_samples = "/blue/qsong1/wang.qing/benchmark_dataset_API/UCE/samples/"
# directory_path_labels = "/blue/qsong1/wang.qing/benchmark_dataset_API/UCE/labels/"

# shape(directory_path_samples,directory_path_labels)