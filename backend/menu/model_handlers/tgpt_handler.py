import os
import json
import csv
import sys
import traceback
from rest_framework.response import Response
from rest_framework import status
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import nn, optim
import pickle
# 修改导入路径，因为LLMs现在在menu目录下
from ..LLMs.tGPT_main.benchmarking_dataloader_EBD import get_dataloaders

def process_tgpt_model(input_dir, output_dir, results):
    """Process files using tGPT model"""
    try:
        # Process the saved files using dataset_generator
        dataset_dirs = dataset_generator(
            directory_path=input_dir,
            output_dir=output_dir,
            is_sorted=True,
            seq_length=8192
        )
        
        # Generate embeddings
        embeddings_dir = generate_embeddings(
            samples_dir=dataset_dirs['samples_dir'],
            output_dir=output_dir
        )

        return Response({
            'message': 'Files processed successfully with tGPT model',
            'files_processed': len(results),
            'input_directory': os.path.relpath(input_dir),
            'output_directory': os.path.relpath(output_dir),
            'samples_dir': os.path.relpath(dataset_dirs['samples_dir']),
            'labels_dir': os.path.relpath(dataset_dirs['labels_dir']),
            'embeddings_directory': os.path.relpath(embeddings_dir),
            'details': results
        }, status=status.HTTP_200_OK)
    except Exception as e:
        raise Exception(f"tGPT processing error: {str(e)}")

def dataset_generator(directory_path, output_dir, is_sorted=True, seq_length=8192, max_len=64):
    """
    Generate datasets from CSV files for tGPT model.
    
    Args:
        directory_path (str): Path to input directory containing CSV files
        output_dir (str): Path to output directory for saving processed files
        is_sorted (bool): Whether to sort the data
        seq_length (int): Length of sequences to generate
        max_len (int): Maximum length for tokenizer
    """
    try:
        # Initialize tokenizer
        tokenizer_file = "lixiangchun/transcriptome-gpt-1024-8-16-64"
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_file)
        
        # Create samples and labels subdirectories
        samples_dir = os.path.join(output_dir, 'tGPT', 'samples')
        labels_dir = os.path.join(output_dir, 'tGPT', 'labels')
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Increase CSV field size limit
        maxInt = sys.maxsize
        while True:
            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt/10)

        files_list = os.listdir(directory_path)
        
        for filename in files_list:
            if not filename.endswith('.csv'):
                continue
                
            labels = []
            samples = []
            csv_file_path = os.path.join(directory_path, filename)
            print('Processing ' + filename)

            with open(csv_file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                pattern = []
                head = True
                
                for row in csv_reader:
                    try:
                        row_data = row[0].split('\t')
                        if head:
                            for gene in row_data:
                                gene = gene.replace('"', '')
                                pattern.append(gene)
                            head = False
                        else:
                            if len(pattern) != len(row_data):
                                print(f"Skipping row due to length mismatch. Expected {len(pattern)}, got {len(row_data)}")
                                continue
                                
                            seq_pattern_order_id_EXPscore = []
                            for i in range(len(row_data)):
                                if i == 0:
                                    pass
                                elif i == 1:
                                    if 'sensitive' in row_data[i].lower():
                                        labels.append(1)
                                    elif 'resistant' in row_data[i].lower():
                                        labels.append(0)
                                else:
                                    seq_pattern_order_id_EXPscore.append((pattern[i], row_data[i]))
                            
                            if is_sorted:
                                seq_pattern_order_id_EXPscore = sorted(seq_pattern_order_id_EXPscore, key=lambda x: float(x[1]), reverse=True)
                            sample = [item[0] for item in seq_pattern_order_id_EXPscore]
                            
                            while len(sample) < seq_length:
                                sample.append(0)
                            sample = sample[:seq_length]
                            _seq = [' '.join(map(str, sample))]
                            seq = tokenizer(_seq, max_length=max_len, truncation=True, padding=True, return_tensors="pt")
                            samples.append(torch.squeeze(seq['input_ids']))
                    except Exception as row_error:
                        print(f"Error processing row in {filename}: {str(row_error)}")
                        continue

            if samples and labels:  # Only save if we have data
                np_samples = np.array(samples)
                np_labels = np.array(labels)
                
                samples_path = os.path.join(samples_dir, f'{filename[:-4]}_samples.npz')
                labels_path = os.path.join(labels_dir, f'{filename[:-4]}_labels.npz')
                
                np.savez(samples_path, array=np_samples)
                np.savez(labels_path, array=np_labels)
                print('Saved ' + filename)
            else:
                print(f"No valid data found in {filename}")

        return {
            'samples_dir': samples_dir,
            'labels_dir': labels_dir
        }
        
    except Exception as e:
        raise Exception(f"Error in dataset generation: {str(e)}")

def finetune_tgpt_model(working_dir, custom_params):
    """Finetune tGPT model"""
    try:
        # TODO: Implement tGPT finetuning
        yield json.dumps({
            'message': 'tGPT finetuning not implemented yet'
        }).encode() + b'\n'
    except Exception as e:
        yield json.dumps({
            'error': f'tGPT finetuning error: {str(e)}'
        }).encode() + b'\n'

class MLP_Classifier(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, num_classes=2):
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

def Accuracy_score(pred, label):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    if pred.device != label.device:
        label = label.to(pred.device)
    acc = (pred == label).float().mean()
    return acc.item()

def F1_score(pred, label):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    if pred.device != label.device:
        label = label.to(pred.device)
    tp = ((pred == 1) & (label == 1)).float().sum()
    fp = ((pred == 1) & (label == 0)).float().sum()
    fn = ((pred == 0) & (label == 1)).float().sum()
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)
    return f1.item()

def AUROC_score(pred, label):
    pred = torch.sigmoid(pred)
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(label, pred)

def Precision_score(pred, label):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    if pred.device != label.device:
        label = label.to(pred.device)
    tp = ((pred == 1) & (label == 1)).float().sum()
    fp = ((pred == 1) & (label == 0)).float().sum()
    precision = tp / (tp + fp + 1e-10)
    return precision.item()

def Recall_score(pred, label):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    if pred.device != label.device:
        label = label.to(pred.device)
    tp = ((pred == 1) & (label == 1)).float().sum()
    fn = ((pred == 0) & (label == 1)).float().sum()
    recall = tp / (tp + fn + 1e-10)
    return recall.item()

def train_fixed_embeddings_tgpt(working_dir, custom_params):
    """Train fixed embeddings for tGPT model"""
    try:
        # Set default parameters
        params = {
            'ep_num': 10,
            'train_batch_size': 128,
            'test_batch_size': 256,
            'lr': 0.0001,
            'train_rate': 0.8
        }
        
        # Update with custom parameters if provided
        if custom_params:
            params.update(custom_params)
        
        # Set paths
        label_path = os.path.join(working_dir, 'tGPT', 'labels')
        data_path = os.path.join(working_dir, 'tGPT', 'embeddings')
        output_path = os.path.join(working_dir, 'tGPT', 'output')
        os.makedirs(output_path, exist_ok=True)

        # Set device and seed
        seed = 24
        torch.manual_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize model
        model = MLP_Classifier().to(device)
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=params['lr'])

        # 创建一个类似 args 的对象来存储参数
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        # 创建 args 对象
        args = Args(
            label_path=label_path,
            data_path=data_path,
            train_batch_size=params['train_batch_size'],
            test_batch_size=params['test_batch_size'],
            train_rate=params['train_rate']
        )

        # 使用 args 对象调用 get_dataloaders
        train_data_loader, test_data_loader = get_dataloaders(args)

        # 初始化指标存储
        final_metrics = {
            'final_train': {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'loss': 0
            },
            'final_test': {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'loss': 0
            },
            'train_loss': [],
            'test_loss': [],
            'epochs': []
        }

        # Training loop
        total_batches = len(train_data_loader)
        for epoch in range(params['ep_num']):
            # 发送 epoch 进度
            yield json.dumps({
                'progress': {
                    'currentEpoch': epoch + 1,
                    'totalEpochs': params['ep_num'],
                    'currentBatch': 0,
                    'totalBatches': total_batches
                }
            }).encode() + b'\n'

            # Training phase
            model.train()
            for batch_idx, _ in enumerate(train_data_loader):
                # 发送 batch 进度
                yield json.dumps({
                    'progress': {
                        'currentEpoch': epoch + 1,
                        'totalEpochs': params['ep_num'],
                        'currentBatch': batch_idx + 1,
                        'totalBatches': total_batches
                    }
                }).encode() + b'\n'

            train_metrics = train_epoch(
                model, train_data_loader, loss_function, 
                optimizer, device, epoch
            )

            # Testing phase
            model.eval()
            test_metrics = test_epoch(
                model, test_data_loader, loss_function, 
                device, epoch
            )

            # 更新历史指标
            final_metrics['epochs'].append(epoch + 1)
            final_metrics['train_loss'].append(train_metrics['loss'])
            final_metrics['test_loss'].append(test_metrics['loss'])

            # 更新最终指标
            final_metrics['final_train'] = {
                'accuracy': train_metrics['accuracy'],
                'precision': train_metrics['precision'],
                'recall': train_metrics['recall'],
                'f1': train_metrics['f1'],
                'loss': train_metrics['loss']
            }
            final_metrics['final_test'] = {
                'accuracy': test_metrics['accuracy'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1': test_metrics['f1'],
                'loss': test_metrics['loss']
            }

            # 发送当前指标
            yield json.dumps({
                'metrics': final_metrics
            }).encode() + b'\n'

        # 保存模型
        model_save_path = os.path.join(output_path, 'model.pth')
        torch.save(model.state_dict(), model_save_path)

        # 发送最终结果
        yield json.dumps({
            'metrics': final_metrics,
            'model_path': model_save_path
        }).encode() + b'\n'

    except Exception as e:
        error_msg = f"tGPT fixed embeddings training error: {str(e)}\nFull traceback:\n{traceback.format_exc()}"
        print(error_msg)
        yield json.dumps({
            'error': error_msg,
            'metrics': {
                'final_train': {
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'loss': 0
                },
                'final_test': {
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'loss': 0
                },
                'train_loss': [],
                'test_loss': [],
                'epochs': []
            }
        }).encode() + b'\n'

def train_epoch(model, dataloader, loss_function, optimizer, device, epoch):
    loss_sum = 0
    pred_all = []
    lbl_all = []
    
    with tqdm(dataloader, desc=f'Training epoch {epoch}') as batches:
        for batch in batches:
            input_embeds, labels = batch
            input_embeds = input_embeds.float().to(device)
            labels = labels.float().to(device)
            
            pred = model(input_embeds)
            loss = loss_function(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            pred_all.extend(pred.detach().cpu())
            lbl_all.extend(labels.cpu())

    pred_all = torch.stack(pred_all)
    lbl_all = torch.stack(lbl_all)
    
    metrics = {
        'loss': loss_sum / len(dataloader),
        'accuracy': Accuracy_score(pred_all, lbl_all),
        'auroc': AUROC_score(pred_all, lbl_all),
        'precision': Precision_score(pred_all, lbl_all),
        'recall': Recall_score(pred_all, lbl_all),
        'f1': F1_score(pred_all, lbl_all)
    }
    
    # 确保所有值都是Python原生类型
    return {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

def test_epoch(model, dataloader, loss_function, device, epoch):
    loss_sum = 0
    pred_all = []
    lbl_all = []
    ebd_all = []
    
    with torch.no_grad():
        with tqdm(dataloader, desc=f'Testing epoch {epoch}') as batches:
            for batch in batches:
                input_embeds, labels = batch
                input_embeds = input_embeds.float().to(device)
                labels = labels.float().to(device)
                pred = model(input_embeds)
                loss = loss_function(pred, labels)

                loss_sum += loss.item()
                pred_all.extend(pred.detach().cpu())
                lbl_all.extend(labels.cpu())
                ebd_all.extend(input_embeds.cpu())

    pred_all = torch.stack(pred_all)
    lbl_all = torch.stack(lbl_all)
    ebd_all = torch.stack(ebd_all)
    
    metrics = {
        'loss': loss_sum / len(dataloader),
        'accuracy': Accuracy_score(pred_all, lbl_all),
        'auroc': AUROC_score(pred_all, lbl_all),
        'precision': Precision_score(pred_all, lbl_all),
        'recall': Recall_score(pred_all, lbl_all),
        'f1': F1_score(pred_all, lbl_all),
        'predictions': pred_all.tolist(),  # 转换为Python列表
        'labels': lbl_all.tolist(),       # 转换为Python列表
        'embeddings': ebd_all.tolist()    # 转换为Python列表
    }
    
    # 确保所有值都是Python原生类型
    return {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

class LineDataset(Dataset):
    def __init__(self, f_path):
        self.lines = np.load(f_path, allow_pickle=True)['array']
        
    def __getitem__(self, i):
        return torch.tensor(self.lines[i])
        
    def __len__(self):
        return len(self.lines)

def generate_embeddings(samples_dir, output_dir, batch_size=64):
    """
    Generate embeddings using the tGPT model.
    
    Args:
        samples_dir (str): Directory containing the sample files
        output_dir (str): Directory to save the embeddings
        batch_size (int): Batch size for processing
    """
    try:
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model and tokenizer
        tokenizer_file = "lixiangchun/transcriptome-gpt-1024-8-16-64"
        checkpoint = "lixiangchun/transcriptome-gpt-1024-8-16-64"
        
        model = GPT2LMHeadModel.from_pretrained(checkpoint, output_hidden_states=True).transformer
        model = model.to(device)
        model.eval()
        
        # Create embeddings directory
        embeddings_dir = os.path.join(output_dir, 'tGPT', 'embeddings')
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Process each file
        files_list = os.listdir(samples_dir)
        for filename in files_list:
            if not filename.endswith('_samples.npz'):
                continue
                
            print(f'Processing {filename} for embeddings')
            
            # Create dataset and dataloader
            ds = LineDataset(os.path.join(samples_dir, filename))
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
            
            Xs = []
            for batch in tqdm(dl, total=len(dl)):
                batch = batch.to(device)
                with torch.no_grad():
                    output = model(batch)
                
                hidden_states = output.last_hidden_state
                mean_states = torch.mean(hidden_states, dim=1).tolist()
                Xs.extend(mean_states)
            
            # Save embeddings
            features = np.stack(Xs)
            output_filename = f"{filename[:-12]}_embeds.npy"
            output_path = os.path.join(embeddings_dir, output_filename)
            np.save(output_path, features)
            print(f'Saved embeddings to {output_path}')
        
        return embeddings_dir
        
    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}") 