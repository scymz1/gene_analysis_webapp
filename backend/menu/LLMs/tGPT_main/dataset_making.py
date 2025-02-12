import csv
import os
from transformers import PreTrainedTokenizerFast
import torch
import numpy as np

def dataset_generator(directory_path, output_dir, is_sorted=True, seq_length=8192, max_len=64):
    """
    Generate datasets from CSV files.
    
    Args:
        directory_path (str): Path to input directory containing CSV files
        output_dir (str): Path to output directory for saving processed files
        is_sorted (bool): Whether to sort the data
        seq_length (int): Length of sequences to generate
        max_len (int): Maximum length for tokenizer
    """
    # Initialize tokenizer
    tokenizer_file = "lixiangchun/transcriptome-gpt-1024-8-16-64"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_file)
    
    # Create samples and labels subdirectories
    samples_dir = os.path.join(output_dir, 'tGPT', 'samples')
    labels_dir = os.path.join(output_dir, 'tGPT', 'labels')
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    csv.field_size_limit(500000000)
    files_list = os.listdir(directory_path)
    
    for filename in files_list:
        if not filename.endswith('.csv'):
            continue
            
        labels = []
        samples = []
        csv_file_path = os.path.join(directory_path, filename)
        print('Processing ' + filename)

        head = True
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            pattern = []
            # sequence level
            for row in csv_reader:
                row_data = row[0].split('\t')
                if head:
                    for gene in row_data:
                        gene = gene.replace('"', '')
                        pattern.append(gene)
                    head = False
                else:
                    if len(pattern)!=len(row_data):
                        print(f"Skipping row due to length mismatch. Expected {len(pattern)}, got {len(row_data)}")
                        continue
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
                            seq_pattern_order_id_EXPscore.append((pattern[i],row_data[i]))
                    
                    if is_sorted:
                        seq_pattern_order_id_EXPscore = sorted(seq_pattern_order_id_EXPscore, key=lambda x: x[1], reverse=True)
                    sample = [item[0] for item in seq_pattern_order_id_EXPscore]
                    
                    while len(sample) < seq_length:
                        sample.append(0)
                    sample = sample[:seq_length]
                    _seq = [' '.join(map(str, sample))]
                    seq = tokenizer(_seq, max_length=max_len, truncation=True, padding=True, return_tensors="pt")
                    samples.append(torch.squeeze(seq['input_ids']))

        np_samples = np.array(samples)
        np_labels = np.array(labels)
        
        samples_path = os.path.join(samples_dir, f'{filename[:-4]}_samples.npz')
        labels_path = os.path.join(labels_dir, f'{filename[:-4]}_labels.npz')
        
        np.savez(samples_path, array=np_samples)
        np.savez(labels_path, array=np_labels)
        print('Saved ' + filename)

    return {
        'samples_dir': samples_dir,
        'labels_dir': labels_dir
    }

# Example usage:
if __name__ == "__main__":
    input_dir = "./"
    output_dir = "./output"
    result = dataset_generator(
        directory_path=input_dir,
        output_dir=output_dir,
        is_sorted=True,
        seq_length=8192
    )




