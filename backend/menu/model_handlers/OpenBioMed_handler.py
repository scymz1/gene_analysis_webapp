import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.join(current_path, "..", "LLMs", "OpenBioMed_main"))
sys.path.append(os.path.join(current_path, "..", "LLMs", "OpenBioMed_main", "open_biomed"))

from open_biomed.datasets.ctc_dataset import Zheng68k
import json
from torch.utils.data import DataLoader

from open_biomed.models.task_model.ctc_model import CTCModel
import torch
import pickle
import tqdm
from rest_framework.response import Response
from rest_framework import status
from ..LLMs.OpenBioMed_main.benchmarking_dataloader_EBD import dataloader
from ..LLMs.OpenBioMed_main.benchmarking_dataloader_FT import dataloader as dataloader_FT
from peft import LoraConfig, get_peft_model
from torch import optim, nn
from ..LLMs.OpenBioMed_main.tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score
import pandas as pd
import csv
import traceback

def process_OpenBioMed_model(input_dir, output_dir, results):
    config = json.load(open(os.path.join(current_path, "..", "LLMs", "OpenBioMed_main", "configs", "ctc", "cellLM.json"), "r"))
    print("load model")
    config["network"]["structure"]["gene2vec_path"] = os.path.join(current_path, "..", "LLMs", "OpenBioMed_main", "assets", "gene2vec_19379_512.npy")
    config["network"]["structure"]["ckpt_path"] = os.path.join(current_path, "..", "LLMs", "OpenBioMed_main", "ckpts", "cell_ckpts", "CellLM.pth")
    model = CTCModel(config["network"], 2)
    files_list = os.listdir(input_dir)

    for filename in files_list:
        print(filename)
        dataset = Zheng68k(path=os.path.join(input_dir, filename), config=config["data"], seed=2023)
        loader = DataLoader(dataset, batch_size=48, shuffle=False)
        print('finished loading')

        with torch.no_grad():
            model.cuda()
            model.eval()
            all_preds, all_y = [], []
            for cell, label in tqdm.tqdm(loader):
                cell = cell.cuda()
                embed = model(cell)

                all_preds.append(embed)
                all_y.append(label)

            all_preds = torch.cat(all_preds, dim=0).cpu()
            all_y = torch.cat(all_y, dim=0).cpu()
        data_dict={}
        data_dict["embeds"]=all_preds
        data_dict["labels"]=all_y
        os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
        with open(os.path.join(output_dir, "samples", filename[:-4]+'_samples.pkl'), 'wb') as pklf:
            pickle.dump(data_dict, pklf)
    return Response({
        'message': 'Files processed successfully',
        'input_directory': os.path.relpath(input_dir),
        'output_directory': os.path.relpath(output_dir),
        'files_processed': len(results),
        'results': results
    }, status=status.HTTP_200_OK)
    
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

def train_fixed_embeddings_OpenBioMed(working_dir, custom_params):
    try:
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        args = Args(
            ep_num=10,
            train_batch_size=128,
            test_batch_size=256,
            data_path=os.path.join(working_dir, "samples"),
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

        train_data_loader, test_data_loader = dataloader(args)


        model = MLP_Classifier().to(device)
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
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
                    input_embeds = input_embeds.to(device)
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
            final_metric['final_train'] = train_metrics
            final_metric['final_test'] = test_metrics
            final_metric['epochs'].append(epoch)
            final_metric['train_loss'].append(train_metrics['loss'])
            final_metric['test_loss'].append(test_metrics['loss'])
            yield json.dumps({
                'metrics': final_metric
            }).encode() + b'\n'
        os.makedirs(os.path.join(working_dir, 'OpenBioMed_model'), exist_ok=True)
        model_save_path = os.path.join(working_dir, 'OpenBioMed_model', 'best_model.pth')        
        torch.save(model.state_dict(), model_save_path)
        yield json.dumps({
            'metrics': final_metric,
            'model_path': model_save_path
        }).encode() + b'\n'
        
    except Exception as e:
        print(f"Error in train_finetune_model: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return Response({
            'message': 'Error in train_finetune_model',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)    

def finetune_OpenBioMed(working_dir, custom_params):
    try:
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)   

        timestamps = working_dir.split('/')[-1]
        input_dir = os.path.join(os.path.dirname(os.path.dirname(working_dir)), "input", timestamps)
        print("input_dir: ", input_dir)
        args = Args(
            ep_num=3,
            train_batch_size=10,
            test_batch_size=10,
            data_path=input_dir,
            lr=0.0001,
            ft_list=['to_q','to_k','to_v'],
            train_rate=0.8
        )

        config = json.load(open(os.path.join(current_path, "..", "LLMs", "OpenBioMed_main", "configs", "ctc", "cellLM.json"), "r"))
        config["network"]["structure"]["gene2vec_path"] = os.path.join(current_path, "..", "LLMs", "OpenBioMed_main", "assets", "gene2vec_19379_512.npy")
        config["network"]["structure"]["ckpt_path"] = os.path.join(current_path, "..", "LLMs", "OpenBioMed_main", "ckpts", "cell_ckpts", "CellLM.pth")
        model = CTCModel(config["network"], 2)

        class FinetuningModel(nn.Module):
            def __init__(self, input_size=512, hidden_size=512, num_classes=2):
                super(FinetuningModel, self).__init__()
                self.CellLMmodel = model
                #print(self.UCEmodel.transformer_encoder)
                # Adding Lora : QKV
                config = LoraConfig(r=8,
                                    lora_alpha=8,
                                    target_modules=['to_q','to_k','to_v'],
                                    lora_dropout=0.05,
                                    bias="none",
                                    task_type="SEQ_CLS")  # [CAUSAL_LM,FEATURE_EXTRACTION,QUESTION_ANS,SEQ_2_SEQ_LM,SEQ_CLS,TOKEN_CLS]

                get_peft_model(self.CellLMmodel, config)  #self.transformer_encoder_lora =

                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, num_classes)


            def forward(self, seq):
                embedding = self.CellLMmodel(seq)
                x = self.fc1(embedding)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.relu(x)
                x = self.fc3(x)
                return x, embedding
        FTmodel = FinetuningModel(input_size=512, hidden_size=512, num_classes=2)   
        total_params = sum(p.numel() for p in FTmodel.parameters()) 
        print(f"Total number of parameters: {total_params}")

        seed = 24
        torch.manual_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        FTmodel = FTmodel.to(device)
        loss_function = torch.nn.BCEWithLogitsLoss()
        model_parameters = [parameter for parameter in FTmodel.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(model_parameters, lr=args.lr)

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
            for batch_idx, (batch_seq, labels) in enumerate(train_data_loader):
                yield json.dumps({
                    'progress': {
                        'currentEpoch': epoch + 1,
                        'totalEpochs': args.ep_num,
                        'currentBatch': batch_idx + 1,
                        'totalBatches': len(train_data_loader)
                    }
                }).encode() + b'\n'
                batch_seq = batch_seq.to(device)
                labels = labels.to(device)
                pred, ebd = FTmodel(batch_seq)
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
                for batch_idx, (batch_seq, labels) in enumerate(test_data_loader):  
                    batch_seq = batch_seq.to(device)
                    labels = labels.to(device)
                    pred, ebd = FTmodel(batch_seq)
                    loss = loss_function(pred, labels)
                    loss_sum = loss_sum + loss
                    pred_all.extend(pred)
                    lbl_all.extend(labels)
                    ebd_all.extend(ebd)

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
            
            final_metric['final_train'] = train_metrics
            final_metric['final_test'] = test_metrics
            final_metric['epochs'].append(epoch)
            final_metric['train_loss'].append(train_metrics['loss'])
            final_metric['test_loss'].append(test_metrics['loss'])
            
            yield json.dumps({
                'metrics': final_metric
            }).encode() + b'\n'
        os.makedirs(os.path.join(working_dir, 'OpenBioMed_model'), exist_ok=True)
        model_save_path = os.path.join(working_dir, 'OpenBioMed_model', 'best_model_FT.pth')        
        torch.save(model.state_dict(), model_save_path)
        yield json.dumps({
            'metrics': final_metric,
            'model_path': model_save_path
        }).encode() + b'\n'
    except Exception as e:
        print(f"Error in finetune_OpenBioMed: {str(e)}")
        print("Traceback:")
        traceback.print_exc()



