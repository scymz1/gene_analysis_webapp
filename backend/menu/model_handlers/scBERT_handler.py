import csv
import os
import pickle
import numpy as np
from rest_framework.response import Response
from rest_framework import status
from pathlib import Path
from typing import List, Dict, Tuple, Any
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from ..LLMs.scBERT_master.performer_pytorch import PerformerLM
from ..LLMs.scBERT_master.utils import *
import torch
from torch import optim, nn
from ..LLMs.scBERT_master.benchmarking_dataloader_EBD import dataloader
from ..LLMs.scBERT_master.benchmarking_dataloader_FT import dataloader as dataloader_FT
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from ..LLMs.scBERT_master.tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score
import pandas as pd
import traceback
import json

def process_header(row_data: List[str], gene_dictionary: Dict[str, Any]) -> List[str]:
    """Process header row and create gene pattern list."""
    pattern = []
    for gene in row_data:
        gene = gene.replace('"', '')
        pattern.append(gene if gene in gene_dictionary else 'none')
    return pattern

def process_data_row(row_data: List[str], pattern: List[str], is_sorted: bool) -> Tuple[List[int], int]:
    """Process a data row and return sample and label."""
    if len(pattern) != len(row_data):
        return None, None
    # Extract label (sensitive=1, resistant=0)
    label = None
    if 'sensitive' in row_data[1]:
        label = 1
    elif 'resistant' in row_data[1]:
        label = 0
    else:
        return None, None
    # Process gene expression scores
    gene_scores = []
    for i, (gene, score) in enumerate(zip(pattern[2:], row_data[2:]), 2):
        if gene != 'none':
            gene_scores.append((gene, score))
    
    if is_sorted:
        gene_scores.sort(key=lambda x: float(x[1]), reverse=True)
    
    return [int(float(item[1])) for item in gene_scores], label

def rebuilder(
    directory_path: str,
    output_dir: str,
    gene_dictionary: Dict[str, Any],
    is_sorted: bool = True,
    seq_length: int = 8192
) -> None:
    """
    Process gene expression data files and save as numpy arrays.
    
    Args:
        directory_path: Path to input directory containing CSV files
        output_dir: Path to output directory for processed files
        gene_dictionary: Dictionary of valid gene names
        is_sorted: Whether to sort gene expressions by value
        seq_length: Maximum sequence length for padding/truncating
    """
    csv.field_size_limit(500000000)
    input_dir = Path(directory_path)
    output_dir = Path(output_dir)
    
    # Create output directories if they don't exist
    (output_dir / 'samples').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels').mkdir(parents=True, exist_ok=True)

    for file_path in input_dir.glob('*.csv'):
        print(f'Processing {file_path.name}')
        labels = []
        samples = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            # Process header
            header_row = next(csv_reader)
            pattern = process_header(header_row[0].split('\t'), gene_dictionary)
            
            # Process data rows
            for row in csv_reader:
                row_data = row[0].split('\t')
                sample, label = process_data_row(row_data, pattern, is_sorted)
                
                if sample is not None and label is not None:
                    # Pad or truncate sample to seq_length
                    sample = sample[:seq_length] + [0] * (seq_length - len(sample))
                    samples.append(sample)
                    labels.append(label)
        
        if samples:
            # Save processed data
            np_samples = np.array(samples)
            np_labels = np.array(labels)
            np.save(output_dir / 'samples' / f'{file_path.stem}_samples.npy', np_samples)
            np.save(output_dir / 'labels' / f'{file_path.stem}_labels.npy', np_labels)
            print(f'{file_path.name} saved')

def process_scBERT_model(input_dir, output_dir, results):
    """Process files using scBERT model"""
    try:
        with open(os.path.join(os.path.dirname(__file__), '..', 'LLMs', 'scBERT_master', 'gene_dict.pkl'), 'rb') as file:
            dictionary = pickle.load(file)
        # Process the saved files using dataset_generator
        rebuilder(input_dir, output_dir, dictionary)
        get_embeddings(output_dir)
        return Response({
            'message': 'Files processed successfully',
            'input_directory': os.path.relpath(input_dir),
            'output_directory': os.path.relpath(output_dir),
            'files_processed': len(results),
            'results': results
        }, status=status.HTTP_200_OK)
    except Exception as e:
        print(f"Error in scBERT processing: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        # Convert results list to dict if it's not already
        if isinstance(results, list):
            results = {'files': results, 'error': str(e)}
        else:
            results['error'] = str(e)
        return Response(results, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



def get_embeddings(working_dir):
    """Get embeddings from scBERT model"""
    SEED = 2021
    EPOCH = 100
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    GRAD_ACC = 60
    VALID_EVERY = 1
    POS_EMBED = True
    # Define missing constants
    CLASS = 7  # Number of token classes
    SEQ_LEN = 8192 + 1  # Maximum sequence length
    
    # Create embeds directory if it doesn't exist
    embeds_dir = os.path.join(working_dir, 'embeds')
    os.makedirs(embeds_dir, exist_ok=True)
    
    DATA_PATH = os.path.join(working_dir, 'samples')
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'LLMs', 'scBERT_master', 'panglao_pretrain.pth')
    CKPT_DIR = os.path.join(working_dir, 'ckpts')
    MODEL_NAME = 'finetune'

    class SCDataset(Dataset):
        def __init__(self, f):
            super().__init__()
            self.data = np.load(f)
            pass

        def __getitem__(self, index):
            #rand_start = random.randint(0, self.data.shape[0]-1)
            full_seq = self.data[index].astype(float)
            full_seq[full_seq > (CLASS - 2)] = CLASS - 2
            full_seq = torch.from_numpy(full_seq).long()
            full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
            return full_seq

        def __len__(self):
            return self.data.shape[0]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_all(SEED)
    model = PerformerLM(
        num_tokens=CLASS,
        dim=200,
        depth=6,
        max_seq_len=SEQ_LEN,
        heads=10,
        local_attn_heads=0,
        g2v_position_emb=POS_EMBED
    )
    ckpt = torch.load(MODEL_PATH)
    model.load_state_dict(ckpt['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    for param in model.norm.parameters():
        param.requires_grad = True
    for param in model.performer.net.layers[-2].parameters():
        param.requires_grad = True
    model = model.to(device)

    files_list = os.listdir(DATA_PATH)
    for file_name in files_list:
        print('processing ' + file_name)
        dataset = SCDataset(os.path.join(DATA_PATH, file_name))
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        pred_all = []
        model.eval()
        pbar = tqdm(loader)
        for data in pbar:
            data = data.to(device)
            pred = model(data)
            pred_all.extend(pred.detach().cpu().numpy())

        pred_all = np.vstack(pred_all)
        np.save(os.path.join(embeds_dir, file_name[:-12] + '_embeds.npy'), pred_all)
        print('saved ' + file_name)
    
class MLP_Classifier(nn.Module):
    def __init__(self, input_size=200, hidden_size=512, num_classes=2):
        super(MLP_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_fixed_embeddings_scBERT(working_dir, custom_params):
    """Train classifier using fixed embeddings from scBERT model"""
    try:
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
                    
        # Set default parameters    
        args = Args(
            ep_num=10,
            train_batch_size=128,
            test_batch_size=128,
            data_path=os.path.join(working_dir, 'embeds'),
            label_path=os.path.join(working_dir, 'labels'),
            lr=0.0001,
            train_rate=0.8,
            ft_list=['none']
        )

        if custom_params:
            for key, value in custom_params.items():
                setattr(args, key, value)

        seed = 24
        torch.manual_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = MLP_Classifier().to(device)
        
        loss_function = torch.nn.BCEWithLogitsLoss()
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(model_parameters, lr=args.lr)

        train_data_loader, test_data_loader = dataloader(args)
        final_metrics = {
            'train_loss': [],
            'test_loss': [],
            'epochs': []
        }
        for epoch in range(args.ep_num):
            loss_sum = 0
            pred_all = []
            lbl_all = []
            yield json.dumps({
                'progress': {
                    'currentEpoch': epoch + 1,
                    'totalEpochs': args.ep_num,
                    'currentBatch': 0,
                    'totalBatches': len(train_data_loader)
                }
            }).encode() + b'\n'
            for batch_idx, (input_embeds, labels) in enumerate(train_data_loader):
                yield json.dumps({
                    'progress': {
                        'currentEpoch': epoch + 1,
                        'totalEpochs': args.ep_num,
                        'currentBatch': batch_idx + 1,
                        'totalBatches': len(train_data_loader)
                    }
                }).encode() + b'\n'
                input_embeds = input_embeds.float().to(device)
                labels = labels.to(device)
                pred = model(input_embeds)
                loss = loss_function(pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum = loss_sum + loss
                pred_all.extend(pred)
                lbl_all.extend(labels)

            pred_all_ = torch.stack(pred_all)
            lbl_all_ = torch.stack(lbl_all)
            train_metrics = {}
            train_metrics['accuracy'] = float(Accuracy_score(pred_all_, lbl_all_))
            train_metrics['f1'] = float(F1_score(pred_all_, lbl_all_))
            try:
                train_metrics['aur'] = float(AUROC_score(pred_all_, lbl_all_))
            except:
                train_metrics['aur'] = 0.0
            train_metrics['precision'] = float(Precision_score(pred_all_, lbl_all_))
            train_metrics['recall'] = float(Recall_score(pred_all_, lbl_all_))
            train_metrics['loss'] = float(loss_sum / len(train_data_loader))

            loss_sum = 0
            pred_all = []
            lbl_all = []
            ebd_all = []
            with torch.no_grad():
                for batch_idx, (input_embeds, labels) in enumerate(test_data_loader):
                    input_embeds = input_embeds.float().to(device)
                    labels = labels.to(device)
                    pred = model(input_embeds)
                    loss = loss_function(pred, labels)
                    loss_sum = loss_sum + loss
                    pred_all.extend(pred)
                    lbl_all.extend(labels)
                    ebd_all.extend(input_embeds)

                pred_all_ = torch.stack(pred_all)
                lbl_all_ = torch.stack(lbl_all)
                ebd_all_ = torch.stack(ebd_all)
                test_metrics = {}
                test_metrics['accuracy'] = float(Accuracy_score(pred_all_, lbl_all_))
                test_metrics['f1'] = float(F1_score(pred_all_, lbl_all_))
                try:
                    test_metrics['aur'] = float(AUROC_score(pred_all_, lbl_all_))
                except:
                    test_metrics['aur'] = 0.0   
                test_metrics['precision'] = float(Precision_score(pred_all_, lbl_all_))
                test_metrics['recall'] = float(Recall_score(pred_all_, lbl_all_))
                test_metrics['loss'] = float(loss_sum / len(test_data_loader))
            final_metrics['final_train'] = train_metrics
            final_metrics['final_test'] = test_metrics
            final_metrics['train_loss'].append(float(train_metrics['loss']))
            final_metrics['test_loss'].append(float(test_metrics['loss']))
            final_metrics['epochs'].append(epoch + 1)
            
            yield json.dumps({
                'metrics': final_metrics
            }).encode() + b'\n'

        model_save_path = os.path.join(working_dir, 'best_model.pth')
        torch.save(model.state_dict(), model_save_path)
        yield json.dumps({
            'metrics': final_metrics,
            'model_path': model_save_path
        }).encode() + b'\n'
        
    except Exception as e:
        print(f"Error in train_fixed_embeddings_scBERT: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return Response({
            'message': 'Error in train_fixed_embeddings_scBERT',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def get_finetune_model(args):
    SEQ_LEN = args.gene_num + 1

    CLASS = args.bin_num + 2
    POS_EMBED_USING = args.pos_embed

    model = PerformerLM(
        num_tokens=CLASS,
        dim=200,
        depth=6,
        max_seq_len=SEQ_LEN,
        heads=10,
        local_attn_heads=0,
        g2v_position_emb=POS_EMBED_USING
    )
    path = args.model_path
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    for param in model.norm.parameters():
        param.requires_grad = True
    for param in model.performer.net.layers[-2].parameters():
        param.requires_grad = True
    #print(model)

    class FinetuningModel(nn.Module):
        def __init__(self, input_size=200, hidden_size=512, num_classes=2):
            super(FinetuningModel, self).__init__()
            self.scBERTmodel = model
            #print(self.UCEmodel.transformer_encoder)
            # Adding Lora : QKV
            config = LoraConfig(r=8,
                                lora_alpha=8,
                                target_modules=args.ft_list,
                                lora_dropout=0.05,
                                bias="none",
                                task_type="SEQ_CLS")  # [CAUSAL_LM,FEATURE_EXTRACTION,QUESTION_ANS,SEQ_2_SEQ_LM,SEQ_CLS,TOKEN_CLS]

            get_peft_model(self.scBERTmodel, config)  #self.transformer_encoder_lora =

            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)


        def forward(self, seq):
            embedding = self.scBERTmodel(seq)

            x = self.fc1(embedding)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x, embedding

    FTmodel = FinetuningModel(input_size=200, hidden_size=512, num_classes=2)
    total_params = sum(p.numel() for p in FTmodel.parameters())
    print(f"Total number of parameters: {total_params}")

    return FTmodel


def finetune_scBERT(working_dir, custom_params):
    """Finetune scBERT model"""
    try:
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)   
                    
        args = Args(
            ep_num=3,
            train_batch_size=3,
            test_batch_size=3,
            data_path=os.path.join(working_dir, 'samples'),
            label_path=os.path.join(working_dir, 'labels'),
            lr=0.0001,
            train_rate=0.8,
            ft_list=['to_k','to_v','to_q'],
            local_rank=-1,
            bin_num=5,
            gene_num=8192,
            epoch=100,
            seed=2021,
            learning_rate=1e-4,
            grad_acc=60,
            pos_embed=True,
            model_path=os.path.join(os.path.dirname(__file__), '..', 'LLMs', 'scBERT_master', 'panglao_pretrain.pth')
        )

        if custom_params:
            for key, value in custom_params.items():
                setattr(args, key, value)
        
        seed = 24
        torch.manual_seed(seed)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = get_finetune_model(args)
        loss_function = torch.nn.BCEWithLogitsLoss()
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(model_parameters, lr=args.lr)

        model = model.to(device)
        train_data_loader, test_data_loader = dataloader_FT(args)
        final_metrics = {
            'train_loss': [],
            'test_loss': [],
            'epochs': []
        }
        for epoch in range(args.ep_num):
            loss_sum = 0
            pred_all = []
            lbl_all = []
            yield json.dumps({
                'progress': {
                    'currentEpoch': epoch + 1,
                    'totalEpochs': args.ep_num,
                    'currentBatch': 0,
                    'totalBatches': len(train_data_loader)
                }
            }).encode() + b'\n'
            for batch_idx, (cell_sentences, labels) in enumerate(train_data_loader):
                yield json.dumps({
                    'progress': {
                        'currentEpoch': epoch + 1,
                        'totalEpochs': args.ep_num,
                        'currentBatch': batch_idx + 1,
                        'totalBatches': len(train_data_loader)
                    }
                }).encode() + b'\n'
                cell_sentences = cell_sentences.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                pred, ebd = model(cell_sentences)
                loss = loss_function(pred, labels)
                loss.backward()
                optimizer.step()
                loss_sum = loss_sum + loss
                pred_all.extend(pred)
                lbl_all.extend(labels)

            pred_all_ = torch.stack(pred_all)
            lbl_all_ = torch.stack(lbl_all)
            train_metrics = {}
            train_metrics['accuracy'] = float(Accuracy_score(pred_all_, lbl_all_))
            train_metrics['f1'] = float(F1_score(pred_all_, lbl_all_))
            try:
                train_metrics['aur'] = float(AUROC_score(pred_all_, lbl_all_))
            except:
                train_metrics['aur'] = 0.0
            train_metrics['precision'] = float(Precision_score(pred_all_, lbl_all_))
            train_metrics['recall'] = float(Recall_score(pred_all_, lbl_all_))
            train_metrics['loss'] = float(loss_sum / len(train_data_loader))

            loss_sum = 0
            pred_all = []
            lbl_all = []
            ebd_all = []
            with torch.no_grad():
                for batch_idx, (cell_sentences, labels) in enumerate(test_data_loader):
                    cell_sentences = cell_sentences.to(device)
                    labels = labels.to(device)
                    pred, ebd = model(cell_sentences)
                    loss = loss_function(pred, labels)
                    loss_sum = loss_sum + loss
                    pred_all.extend(pred)
                    lbl_all.extend(labels)
                    ebd_all.extend(ebd)
                test_metrics = {}
                test_metrics['accuracy'] = float(Accuracy_score(pred_all_, lbl_all_))
                test_metrics['f1'] = float(F1_score(pred_all_, lbl_all_))
                try:
                    test_metrics['aur'] = float(AUROC_score(pred_all_, lbl_all_))
                except:
                    test_metrics['aur'] = 0.0
                test_metrics['precision'] = float(Precision_score(pred_all_, lbl_all_))
                test_metrics['recall'] = float(Recall_score(pred_all_, lbl_all_))
                test_metrics['loss'] = float(loss_sum / len(test_data_loader))
            final_metrics['final_train'] = train_metrics
            final_metrics['final_test'] = test_metrics
            final_metrics['train_loss'].append(float(train_metrics['loss']))
            final_metrics['test_loss'].append(float(test_metrics['loss']))
            final_metrics['epochs'].append(epoch + 1)
            
            yield json.dumps({
                'metrics': final_metrics
            }).encode() + b'\n'
            
        model_save_path = os.path.join(working_dir, 'best_model.pth')
        torch.save(model.state_dict(), model_save_path)
        yield json.dumps({
            'metrics': final_metrics,
            'model_path': model_save_path
        }).encode() + b'\n'
        
    except Exception as e:
        print(f"Error in finetune_scBERT: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return Response({
            'message': 'Error in finetune_scBERT',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
                