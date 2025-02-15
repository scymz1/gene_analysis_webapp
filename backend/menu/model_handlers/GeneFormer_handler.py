import csv
import os
import pickle
import numpy as np
import torch
from transformers import BertForTokenClassification,BertForSequenceClassification
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from ..LLMs.Geneformer_finetuing_lora_prompt_cell_cls.embeds_loader import embeds
from rest_framework.response import Response
from rest_framework import status
from ..LLMs.Geneformer_finetuing_lora_prompt_cell_cls.benchmarking_dataloader_EBD import dataloader
from ..LLMs.Geneformer_finetuing_lora_prompt_cell_cls.tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score
import pandas as pd
import json
import traceback
from ..LLMs.Geneformer_finetuing_lora_prompt_cell_cls.benchmarking_dataloader_FT import dataloader as dataloader_FT
from peft import LoraConfig, get_peft_model
pretrained_model_name = "geneformer-6L-30M_CellClassifier_cardiomyopathies_220224/"

def process_GeneFormer_model(input_dir, output_dir, results):
    with open(os.path.join(os.path.dirname(__file__), "..", "LLMs", "Geneformer_finetuing_lora_prompt_cell_cls", "my_dict.pkl"), 'rb') as file:
        dictionary = pickle.load(file)
    csv.field_size_limit(500000000)
    files_list = os.listdir(input_dir)
    is_sorted = True
    seq_length = 2048
    for filename in files_list:
        csv_file_path = os.path.join(input_dir, filename)
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

        os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        # datset save path
        file_path_samples = os.path.join(output_dir, 'samples', filename[:-4] + '_samples.npy')
        file_path_labels = os.path.join(output_dir, 'labels', filename[:-4] + '_labels.npy')

        np.save(file_path_samples,np.array(samples))
        np.save(file_path_labels,np.array(labels))

        get_ebd(output_dir)
        return Response({
            'message': 'Files processed successfully',
            'input_directory': os.path.relpath(input_dir),
            'output_directory': os.path.relpath(output_dir),
            'files_processed': len(results),
            'results': results
        }, status=status.HTTP_200_OK)

def get_ebd(output_dir):
    class geneformer(nn.Module):
        def __init__(self):
            pretrained_model_path = os.path.join(os.path.dirname(__file__), "..", "LLMs", "Geneformer_finetuing_lora_prompt_cell_cls", "geneformer-6L-30M_CellClassifier_cardiomyopathies_220224")
            super(geneformer, self).__init__()
            self.former = BertForSequenceClassification.from_pretrained(
            pretrained_model_path,
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

    directory_path = os.path.join(output_dir, 'samples')
    dt_list = os.listdir(directory_path)
    for dt in dt_list:
        print('Processing: '+ dt)
        dataset = embeds(data_path=os.path.join(directory_path, dt))
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False,num_workers=0)
        pbar = tqdm(dataloader)
        dataset_embeds = []
        with torch.no_grad():
            for batch in pbar:
                my_tensor = torch.tensor(batch).to(device)  
                op = model(my_tensor.long())
                dataset_embeds.append(op.detach().cpu().numpy())

        dataset_embeds = np.vstack(dataset_embeds)
        os.makedirs(os.path.join(output_dir, 'embeds'), exist_ok=True)
        np.save(os.path.join(output_dir, 'embeds', dt[:-4]+'_embeds.npy'), dataset_embeds)


def train_fixed_embeddings_GeneFormer(working_dir, custom_params):
    try:
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        args = Args(
            ep_num=10,
            train_batch_size=128,
            test_batch_size=256,
            data_path=os.path.join(working_dir, "embeds"),
            label_path=os.path.join(working_dir, "labels"),
            lr=0.0001,
            train_rate=0.8,
            ft_list=['none']
        )

        if custom_params:
            for key, value in custom_params.items():
                setattr(args, key, value)

        class MLP_Classifier(nn.Module):
            def __init__(self, input_size=256, hidden_size=512, num_classes=2):
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
                
        seed = 24
        torch.manual_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = MLP_Classifier()
        model = model.to(device)

        loss_function = torch.nn.BCEWithLogitsLoss()
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(model_parameters, lr=args.lr)

        train_data_loader, test_data_loader = dataloader(args)
        final_metrics = {
            'train_loss': [],
            'test_loss': [],
            'epochs': []
        }
        print('Training GeneFormer model...')
        for epoch in range(args.ep_num):
            loss_sum = 0
            pred_all = []
            lbl_all = []
            y_pred = []
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
                input_embeds = input_embeds.to(device)
                labels = labels.to(device)
                pred = model(input_embeds)
                loss = loss_function(pred, labels)
                loss_sum += loss.item()
                pred_all.extend(pred)
                lbl_all.extend(labels)
            
            pred_all_ = torch.stack(pred_all)
            lbl_all_ = torch.stack(lbl_all)
            train_metrics = {}
            train_metrics['loss'] = loss_sum/len(train_data_loader)
            train_metrics['accuracy'] = Accuracy_score(pred_all_, lbl_all_)
            train_metrics['f1'] = F1_score(pred_all_, lbl_all_)
            try:
                train_metrics['aur'] = AUROC_score(pred_all_, lbl_all_)
            except:
                train_metrics['aur'] = 0.0   
            train_metrics['precision'] = Precision_score(pred_all_, lbl_all_)
            train_metrics['recall'] = Recall_score(pred_all_, lbl_all_)

            loss_sum = 0
            pred_all = []
            lbl_all = []
            with torch.no_grad():
                for batch_idx, (input_embeds, labels) in enumerate(test_data_loader):
                    input_embeds = input_embeds.to(device)
                    labels = labels.to(device)  
                    pred = model(input_embeds)
                    loss = loss_function(pred, labels)
                    loss_sum += loss.item()
                    pred_all.extend(pred)
                    lbl_all.extend(labels)      
                    
                pred_all_ = torch.stack(pred_all)
                lbl_all_ = torch.stack(lbl_all)
                test_metrics = {}
                test_metrics['loss'] = loss_sum/len(test_data_loader)
                test_metrics['accuracy'] = Accuracy_score(pred_all_, lbl_all_)
                test_metrics['f1'] = F1_score(pred_all_, lbl_all_)
                try:
                    test_metrics['aur'] = AUROC_score(pred_all_, lbl_all_)
                except:
                    test_metrics['aur'] = 0.0   
                test_metrics['precision'] = Precision_score(pred_all_, lbl_all_)
                test_metrics['recall'] = Recall_score(pred_all_, lbl_all_)
            final_metrics['final_train'] = train_metrics
            final_metrics['final_test'] = test_metrics
            final_metrics['epochs'].append(epoch)
            final_metrics['train_loss'].append(train_metrics['loss'])
            final_metrics['test_loss'].append(test_metrics['loss'])

            yield json.dumps({
                'metrics': final_metrics
            }).encode() + b'\n'
        os.makedirs(os.path.join(working_dir, 'GeneFormer_model'), exist_ok=True)
        model_save_path = os.path.join(working_dir, 'GeneFormer_model', 'best_model.pth')        
        torch.save(model.state_dict(), model_save_path)
        yield json.dumps({
            'metrics': final_metrics,
            'model_path': model_save_path
        }).encode() + b'\n'

    except Exception as e:
        print(f"Error in train_fixed_embeddings_GeneFormer: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        yield json.dumps({
            'error': str(e)
        }).encode() + b'\n'

def finetune_GeneFormer(working_dir, custom_params):
    try:
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        args = Args(
            ep_num=3,
            train_batch_size=10,
            test_batch_size=10,
            data_path=os.path.join(working_dir, "samples"),
            label_path=os.path.join(working_dir, "labels"), 
            lr=0.0001,
            train_rate=0.8,
            ft_list=['query', 'key', 'value']
        )
        
        if custom_params:
            for key, value in custom_params.items():
                setattr(args, key, value)

        seed = 24
        torch.manual_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        pretrained_model_path = os.path.join(os.path.dirname(__file__), "..", "LLMs", "Geneformer_finetuing_lora_prompt_cell_cls", "geneformer-6L-30M_CellClassifier_cardiomyopathies_220224")
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model_path,
            num_labels=3,
            output_attentions=False,
            output_hidden_states=True
        )


        class FinetuningModel(nn.Module):
            def __init__(self, input_size=256, hidden_size=512, num_classes=2):
                super(FinetuningModel, self).__init__()
                self.Geneformermodel = model
                # Adding Lora : QKV
                config = LoraConfig(r=8,
                                    lora_alpha=8,
                                    target_modules=args.ft_list,
                                    lora_dropout=0.05,
                                    bias="none",
                                    task_type="SEQ_CLS")  # [CAUSAL_LM,FEATURE_EXTRACTION,QUESTION_ANS,SEQ_2_SEQ_LM,SEQ_CLS,TOKEN_CLS]

                get_peft_model(self.Geneformermodel, config)  #self.transformer_encoder_lora =

                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, num_classes)


            def forward(self, seq):
                output = self.Geneformermodel(seq)
                hidden_states = output['hidden_states'][6]
                ebd = torch.mean(hidden_states, dim=1)
                x = self.fc1(ebd)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.relu(x)
                x = self.fc3(x)
                return x,ebd

        FTmodel = FinetuningModel(input_size=256, hidden_size=512, num_classes=2)
        total_params = sum(p.numel() for p in FTmodel.parameters())
        print(f"Total number of parameters: {total_params}")

        loss_function = torch.nn.BCEWithLogitsLoss()
        model_parameters = [parameter for parameter in FTmodel.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(model_parameters, lr=args.lr)

        FTmodel = FTmodel.to(device)
        train_data_loader, test_data_loader = dataloader_FT(args)   
        final_metrics = {
            'train_loss': [],
            'test_loss': [],
            'epochs': []
        }
        # print('Training GeneFormer model...')
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
            for batch_idx, (seq, labels) in enumerate(train_data_loader):
                yield json.dumps({
                    'progress': {
                        'currentEpoch': epoch + 1,
                        'totalEpochs': args.ep_num,
                        'currentBatch': batch_idx + 1,
                        'totalBatches': len(train_data_loader)
                    }
                }).encode() + b'\n'
                my_tensor = torch.tensor(seq).to(device)
                labels = labels.to(device)
                pred, ebd = FTmodel(my_tensor.long())
                loss = loss_function(pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                pred_all.extend(pred)
                lbl_all.extend(labels)

            pred_all_ = torch.stack(pred_all)
            lbl_all_ = torch.stack(lbl_all)
            train_metrics = {}
            train_metrics['loss'] = loss_sum/len(train_data_loader)
            train_metrics['accuracy'] = Accuracy_score(pred_all_, lbl_all_)
            train_metrics['f1'] = F1_score(pred_all_, lbl_all_)
            try:
                train_metrics['aur'] = AUROC_score(pred_all_, lbl_all_)
            except:
                train_metrics['aur'] = 0.0
            train_metrics['precision'] = Precision_score(pred_all_, lbl_all_)
            train_metrics['recall'] = Recall_score(pred_all_, lbl_all_)

            loss_sum = 0
            pred_all = []
            lbl_all = []
            ebd_all = []
            with torch.no_grad():
                for batch_idx, (seq, labels) in enumerate(test_data_loader):
                    my_tensor = torch.tensor(seq).to(device)
                    labels = labels.to(device)
                    pred, ebd = FTmodel(my_tensor.long())
                    loss = loss_function(pred, labels)
                    loss_sum += loss.item()
                    pred_all.extend(pred)   
                    lbl_all.extend(labels)
                    ebd_all.extend(ebd)

                pred_all_ = torch.stack(pred_all)
                lbl_all_ = torch.stack(lbl_all)
                ebd_all_ = torch.stack(ebd_all)     
                test_metrics = {}
                test_metrics['loss'] = loss_sum/len(test_data_loader)
                test_metrics['accuracy'] = Accuracy_score(pred_all_, lbl_all_)
                test_metrics['f1'] = F1_score(pred_all_, lbl_all_)
                try:
                    test_metrics['aur'] = AUROC_score(pred_all_, lbl_all_)      
                except:
                    test_metrics['aur'] = 0.0
                test_metrics['precision'] = Precision_score(pred_all_, lbl_all_)
                test_metrics['recall'] = Recall_score(pred_all_, lbl_all_)

            final_metrics['final_train'] = train_metrics
            final_metrics['final_test'] = test_metrics
            final_metrics['epochs'].append(epoch)
            final_metrics['train_loss'].append(train_metrics['loss'])
            final_metrics['test_loss'].append(test_metrics['loss'])

            yield json.dumps({
                'metrics': final_metrics
            }).encode() + b'\n' 
        os.makedirs(os.path.join(working_dir, 'GeneFormer_model_FT'), exist_ok=True)
        model_save_path = os.path.join(working_dir, 'GeneFormer_model_FT', 'best_model.pth')        
        torch.save(FTmodel.state_dict(), model_save_path)
        yield json.dumps({
            'metrics': final_metrics,
            'model_path': model_save_path   
        }).encode() + b'\n'

    except Exception as e:
        print(f"Error in finetune_GeneFormer: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        yield json.dumps({
            'error': str(e)
        }).encode() + b'\n'