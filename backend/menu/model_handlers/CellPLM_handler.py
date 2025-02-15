import warnings
warnings.filterwarnings("ignore")
import numpy as np

from ..LLMs.CellPLM_main.CellPLM.utils import set_seed
from ..LLMs.CellPLM_main.CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
import os
import csv
import json
import torch
import pickle
import mygene
from rest_framework.response import Response
from rest_framework import status
import traceback
from ..LLMs.CellPLM_main.benchmarking_dataloader_EBD import dataloader
from tqdm import tqdm
from transformers import BertTokenizer
from torch import optim
from ..LLMs.CellPLM_main.tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score
import pandas as pd
from torch import nn
from ..LLMs.CellPLM_main.benchmarking_dataloader_FT import dataloader as dataloader_FT
from ..LLMs.CellPLM_main.CellPLM.pipeline.cell_embed_ft import CellEmbeddingPipeline
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict


def process_CellPLM_model(input_dir, output_dir, results):
    try:
        with open(os.path.join(os.path.dirname(__file__), "..", "LLMs", "CellPLM_main", "ckpt", "20230926_85M.config.json"), 'r') as file:
            data = json.load(file)

        model_gene_set = set(data['gene_list'])
        def symbol_to_ensembl(gene_list):
            mg = mygene.MyGeneInfo()
            return mg.querymany(gene_list, scopes='symbol', fields='ensembl.gene', as_dataframe=True,
                        species='human').reset_index().drop_duplicates(subset='query')['ensembl.gene'].fillna('0').tolist()
        
        set_seed(42)

        csv.field_size_limit(500000000)
        files_list = os.listdir(input_dir)

        for filename in files_list:
            labels = []
            samples = []
            pattern = []
            dt = {}
            csv_file_path = os.path.join(input_dir, filename)
            print('processing ' + csv_file_path)
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
            os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
            with open(os.path.join(output_dir, 'samples', filename[:-4] + '_samples.pkl'), "wb") as f:
                pickle.dump(adata, f)

            os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
            np.save(os.path.join(output_dir, 'labels', filename[:-4] + '_labels.npy'), labels)
            print('saved ' + csv_file_path)
            get_embeddings(input_dir, output_dir, results)
            return Response({
                'message': 'Files processed successfully',
                'input_directory': os.path.relpath(input_dir),
                'output_directory': os.path.relpath(output_dir),
                'files_processed': len(results),
                'results': results
            }, status=status.HTTP_200_OK)
    except Exception as e:
        print(f"Error in process_cellPLM_data: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def get_embeddings(input_dir, output_dir, results):
    try:
        os.makedirs(os.path.join(output_dir, 'embeds'), exist_ok=True)
        with open(os.path.join(os.path.dirname(__file__), "..", "LLMs", "CellPLM_main", "ckpt", "20230926_85M.config.json"), 'r') as file:
            data = json.load(file)

        model_gene_set = set(data['gene_list'])
        def symbol_to_ensembl(gene_list):
            mg = mygene.MyGeneInfo()
            return mg.querymany(gene_list, scopes='symbol', fields='ensembl.gene', as_dataframe=True,
                        species='human').reset_index().drop_duplicates(subset='query')['ensembl.gene'].fillna('0').tolist()

        PRETRAIN_VERSION = '20230926_85M'
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        set_seed(42)

        csv.field_size_limit(5000000)
        files_list = os.listdir(input_dir)
        pipeline = CellEmbeddingPipeline(pretrain_prefix=PRETRAIN_VERSION,  # Specify the pretrain checkpoint to load
                                        pretrain_directory=os.path.join(os.path.dirname(__file__), "..", "LLMs", "CellPLM_main", "ckpt"))

        for filename in files_list:
            labels = []
            samples = []
            pattern = []
            dt = {}
            csv_file_path = os.path.join(input_dir, filename)
            print('processing '+ csv_file_path)
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
            embedding = pipeline.predict(adata,  # An AnnData object
                                        device=DEVICE)  # Specify a gpu or cpu for model inference

            dataset_embeds = embedding.cpu().numpy()
            np.save(os.path.join(output_dir, 'embeds', filename[:-4] + '_embeds.npy'), dataset_embeds)
            np.save(os.path.join(output_dir, 'labels', filename[:-4] + '_labels.npy'), labels)
            print('saved '+ csv_file_path)
    except Exception as e:
        print(f"Error in get_embeddings: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

class MLP_Classifier(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_classes=2):
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

def train_fixed_embeddings_CellPLM(working_dir, custom_params):
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

        seed = 24
        torch.manual_seed(seed) 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = MLP_Classifier().to(device)

        loss_function = torch.nn.BCEWithLogitsLoss()
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(model_parameters, lr=args.lr)

        train_data_loader, test_data_loader = dataloader(args)
        final_metric = {
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
                input_embeds = input_embeds.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                pred = model(input_embeds)
                loss = loss_function(pred, labels)
                loss.backward()
                optimizer.step()

                loss_sum = loss_sum + loss
                pred_all.extend(pred)
                lbl_all.extend(labels)
            
            pred_all_ = torch.stack(pred_all)
            lbl_all_ = torch.stack(lbl_all) 
            train_metrics = {}
            train_metrics['loss'] = float(loss_sum / len(train_data_loader))
            train_metrics['accuracy'] = float(Accuracy_score(pred_all_, lbl_all_))
            train_metrics['f1'] = float(F1_score(pred_all_, lbl_all_))
            try:
                train_metrics['aur'] = float(AUROC_score(pred_all_, lbl_all_))
            except:
                train_metrics['aur'] = 0.0
            train_metrics['precision'] = float(Precision_score(pred_all_, lbl_all_))    
            train_metrics['recall'] = float(Recall_score(pred_all_, lbl_all_))

            loss_sum = 0
            pred_all = []
            lbl_all = []
            with torch.no_grad():
                for batch_idx, (input_embeds, labels) in enumerate(test_data_loader)    :
                    input_embeds = input_embeds.to(device)
                    labels = labels.to(device)
                    pred = model(input_embeds)
                    loss = loss_function(pred, labels)
                    loss_sum = loss_sum + loss
                    pred_all.extend(pred)
                    lbl_all.extend(labels)

                pred_all_ = torch.stack(pred_all)
                lbl_all_ = torch.stack(lbl_all)
                test_metrics = {}
                test_metrics['loss'] = float(loss_sum / len(test_data_loader))
                test_metrics['accuracy'] = float(Accuracy_score(pred_all_, lbl_all_))
                test_metrics['f1'] = float(F1_score(pred_all_, lbl_all_))
                try:
                    test_metrics['aur'] = float(AUROC_score(pred_all_, lbl_all_))
                except:
                    test_metrics['aur'] = 0.0
                test_metrics['precision'] = float(Precision_score(pred_all_, lbl_all_)) 
                test_metrics['recall'] = float(Recall_score(pred_all_, lbl_all_))

            final_metric['train_loss'].append(train_metrics['loss'])
            final_metric['test_loss'].append(test_metrics['loss'])
            final_metric['epochs'].append(epoch)
            final_metric['final_train'] = train_metrics
            final_metric['final_test'] = test_metrics
            yield json.dumps({
                'metrics': final_metric
            }).encode() + b'\n' 
        os.makedirs(os.path.join(working_dir, 'CellPLM_model'), exist_ok=True)
        model_save_path = os.path.join(working_dir, 'CellPLM_model', 'best_model.pth')        
        torch.save(model.state_dict(), model_save_path)
        yield json.dumps({
            'metrics': final_metric,
            'model_path': model_save_path
        }).encode() + b'\n' 
    except Exception as e:
        print(f"Error in get_filtered_embeddings: {str(e)}")
        print("Traceback:")
        traceback.print_exc()   

def finetune_CellPLM(working_dir, custom_params):
    try:
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        args = Args(
            ep_num=3,
            train_batch_size=1,
            test_batch_size=1,
            data_path=os.path.join(working_dir, "samples"),
            label_path=os.path.join(working_dir, "labels"),
            lr=0.0001,
            train_rate=0.8,
            ft_list=['query_projection','key_projection','value_projection'],
            max_length=2048,
            pad_length=1536,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        if custom_params:
            for key, value in custom_params.items():
                setattr(args, key, value)
        
        seed = 24
        torch.manual_seed(seed) 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
        PRETRAIN_VERSION = '20230926_85M'
        pipeline = CellEmbeddingPipeline(pretrain_prefix=PRETRAIN_VERSION,  # Specify the pretrain checkpoint to load
                                        pretrain_directory=os.path.join(os.path.dirname(__file__), "..", "LLMs", "CellPLM_main", "ckpt"))
        model = pipeline.get_model()  # Specify a gpu or cpu for model inference
        model.to(args.device)   

        class FinetuningModel(nn.Module):
            def __init__(self, input_size=512, hidden_size=512, num_classes=2):
                super(FinetuningModel, self).__init__()
                self.CellPLMmodel = model
                #print(self.UCEmodel.transformer_encoder)
                # Adding Lora : QKV
                config = LoraConfig(r=8,
                                    lora_alpha=8,
                                    target_modules=args.ft_list,
                                    lora_dropout=0.05,
                                    bias="none",
                                    task_type="SEQ_CLS")  # [CAUSAL_LM,FEATURE_EXTRACTION,QUESTION_ANS,SEQ_2_SEQ_LM,SEQ_CLS,TOKEN_CLS]

                get_peft_model(self.CellPLMmodel, config)  #self.transformer_encoder_lora =
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, num_classes)


            def forward(self, x_seq, gene_list):
                new = {}
                new['x_seq'] = x_seq
                x_dict = new
                out_dict, _ = self.CellPLMmodel(x_dict, gene_list)
                embedding = out_dict['pred']  # [input_dict['order_list']])
                x = self.fc1(embedding)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.relu(x)
                x = self.fc3(x)
                return x, embedding
                
        FTmodel = FinetuningModel(input_size=512, hidden_size=512, num_classes=2)
        total_params = sum(p.numel() for p in FTmodel.parameters())
        print(f"Total number of parameters: {total_params}")

        loss_function = torch.nn.BCEWithLogitsLoss()
        model_parameters = [parameter for parameter in FTmodel.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(model_parameters, lr=args.lr)
        FTmodel = FTmodel.to(args.device)
        train_data_loader, test_data_loader = dataloader_FT(args)
        final_metric = {
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
            for batch_idx, (x_seq, gene_list, labels) in enumerate(train_data_loader):
                yield json.dumps({
                    'progress': {
                        'currentEpoch': epoch + 1,
                        'totalEpochs': args.ep_num,
                        'currentBatch': batch_idx + 1,
                        'totalBatches': len(train_data_loader)
                    }
                }).encode() + b'\n'
                x_seq = x_seq.to(args.device)
                gene_list = [item[0] for item in gene_list]
                labels = labels.to(args.device)
                optimizer.zero_grad()
                pred, ebd = FTmodel(x_seq, gene_list)
                loss = loss_function(pred, labels)
                loss.backward()
                optimizer.step()

                loss_sum = loss_sum + loss
                pred_all.extend(pred)
                lbl_all.extend(labels)  
                
            pred_all_ = torch.stack(pred_all)
            lbl_all_ = torch.stack(lbl_all)
            train_metrics = {}
            train_metrics['loss'] = float(loss_sum / len(train_data_loader))
            train_metrics['accuracy'] = float(Accuracy_score(pred_all_, lbl_all_))
            train_metrics['f1'] = float(F1_score(pred_all_, lbl_all_))
            try:
                train_metrics['aur'] = float(AUROC_score(pred_all_, lbl_all_))
            except:
                train_metrics['aur'] = 0.0
            train_metrics['precision'] = float(Precision_score(pred_all_, lbl_all_))
            train_metrics['recall'] = float(Recall_score(pred_all_, lbl_all_))

            loss_sum = 0
            pred_all = []
            lbl_all = []
            ebd_all = []
            with torch.no_grad():
                for batch_idx, (x_seq, gene_list, labels) in enumerate(test_data_loader):
                    x_seq = x_seq.to(args.device)
                    gene_list = [item[0] for item in gene_list]
                    labels = labels.to(args.device)
                    pred, ebd = FTmodel(x_seq, gene_list)
                    loss = loss_function(pred, labels)
                    loss_sum = loss_sum + loss
                    pred_all.extend(pred)
                    lbl_all.extend(labels)
                    ebd_all.extend(ebd)

                pred_all_ = torch.stack(pred_all)
                lbl_all_ = torch.stack(lbl_all)
                ebd_all_ = torch.stack(ebd_all)
                test_metrics = {}
                test_metrics['loss'] = float(loss_sum / len(test_data_loader))
                test_metrics['accuracy'] = float(Accuracy_score(pred_all_, lbl_all_))
                test_metrics['f1'] = float(F1_score(pred_all_, lbl_all_))
                try:
                    test_metrics['aur'] = float(AUROC_score(pred_all_, lbl_all_))   
                except:
                    test_metrics['aur'] = 0.0
                test_metrics['precision'] = float(Precision_score(pred_all_, lbl_all_))
                test_metrics['recall'] = float(Recall_score(pred_all_, lbl_all_))

            final_metric['train_loss'].append(train_metrics['loss'])
            final_metric['test_loss'].append(test_metrics['loss'])
            final_metric['epochs'].append(epoch)
            final_metric['final_train'] = train_metrics
            final_metric['final_test'] = test_metrics
            yield json.dumps({
                'metrics': final_metric
            }).encode() + b'\n'
        os.makedirs(os.path.join(working_dir, 'CellPLM_model_FT'), exist_ok=True)
        model_save_path = os.path.join(working_dir, 'CellPLM_model_FT', 'best_model.pth')        
        torch.save(FTmodel.state_dict(), model_save_path)
        yield json.dumps({
            'metrics': final_metric,
            'model_path': model_save_path
        }).encode() + b'\n'
    except Exception as e:
        print(f"Error in finetune_CellPLM: {str(e)}")
        print("Traceback:")
        traceback.print_exc()