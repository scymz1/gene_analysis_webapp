from ..LLMs.scGPT_main.scgpt.tasks.get_ebd import embed_data
import os
import traceback
from rest_framework.response import Response
from rest_framework import status
import torch
from ..LLMs.scGPT_main.scgpt.tasks.benchmarking_main_EBD import dataloader
from tqdm import tqdm
from torch import optim
from ..LLMs.scGPT_main.scgpt.tasks.tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score
import pandas as pd
import pickle
from torch import nn
import csv
import json

from ..LLMs.scGPT_main.scgpt.model import TransformerModel
from ..LLMs.scGPT_main.scgpt.tasks.benchmarking_dataloader_FT import dataloader as dataloader_FT
from ..LLMs.scGPT_main.scgpt.tokenizer import GeneVocab
from peft import LoraConfig, get_peft_model
from ..LLMs.scGPT_main.scgpt.tasks.utils import load_pretrained

def process_scGPT_model(input_dir, output_dir, results):
    try:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'LLMs', 'scGPT_main', 'scgpt', 'save')
        for file_name in os.listdir(input_dir):
            embed_data(adata_or_file=os.path.join(input_dir, file_name), 
            model_dir=model_dir,
            use_fast_transformer=True,
            max_length=1200,
            batch_size=64,
            gene_col = "index",
            filename = file_name,
            output_dir=output_dir)
        return Response({
            'message': 'Files processed successfully',
            'input_directory': os.path.relpath(input_dir),
            'output_directory': os.path.relpath(output_dir),
            'files_processed': len(results),
            'results': results
        }, status=status.HTTP_200_OK)
    except Exception as e:
        print(f"Error in scFoundation processing: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return Response({
            'error': f'Error in scFoundation processing: {str(e)}',
            'files_were_saved': True,
            'input_directory': input_dir
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def train_fixed_embeddings_scGPT(working_dir, custom_params):
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
                optimizer.zero_grad()
                pred = model(input_embeds)
                loss = loss_function(pred, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss
                pred_all.extend(pred)
                lbl_all.extend(labels)
            pred_all_ = torch.stack(pred_all)
            lbl_all_ = torch.stack(lbl_all) 
            train_metrics = {}
            train_metrics['loss'] = float(loss_sum/len(train_data_loader))
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
                for batch_idx, (input_embeds, labels) in enumerate(test_data_loader):
                    input_embeds = input_embeds.float().to(device)
                    labels = labels.to(device)
                    pred = model(input_embeds)
                    loss = loss_function(pred, labels)
                    loss_sum += loss
                    pred_all.extend(pred)
                    lbl_all.extend(labels)
                    ebd_all.extend(input_embeds.to('cpu'))
                pred_all_ = torch.stack(pred_all)
                lbl_all_ = torch.stack(lbl_all)
                ebd_all_ = torch.stack(ebd_all)
                test_metrics = {}
                test_metrics['loss'] = float(loss_sum/len(test_data_loader))
                test_metrics['accuracy'] = float(Accuracy_score(pred_all_, lbl_all_))
                try:
                    test_metrics['aur'] = float(AUROC_score(pred_all_, lbl_all_))
                except:
                    test_metrics['aur'] = 0.0
                test_metrics['precision'] = float(Precision_score(pred_all_, lbl_all_))
                test_metrics['recall'] = float(Recall_score(pred_all_, lbl_all_))
                test_metrics['f1'] = float(F1_score(pred_all_, lbl_all_))
            final_metrics['final_train'] = train_metrics    
            final_metrics['final_test'] = test_metrics
            final_metrics['epochs'].append(epoch)
            final_metrics['train_loss'].append(train_metrics['loss'])
            final_metrics['test_loss'].append(test_metrics['loss'])
            yield json.dumps({
                'metrics': final_metrics
            }).encode() + b'\n' 
        os.makedirs(os.path.join(working_dir, 'scGPT_model'), exist_ok=True)
        model_save_path = os.path.join(working_dir, 'scGPT_model', 'best_model.pth')
        torch.save(model.state_dict(), model_save_path)
        yield json.dumps({
            'metrics': final_metrics,
            'model_path': model_save_path
        }).encode() + b'\n'
    except Exception as e:
        print(f"Error in scGPT training: {str(e)}")
        print("Traceback:")
        traceback.print_exc()


def finetune_scGPT(working_dir, custom_params):
    try:
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        timestamps = working_dir.split('/')[-1]
        input_dir = os.path.join(os.path.dirname(os.path.dirname(working_dir)), "input", timestamps)
        args = Args(
            ep_num=3,
            train_batch_size=10,
            test_batch_size=10,
            lr=0.0001,
            data_path=input_dir,
            train_rate=0.8,
            ft_list=['out_proj'],
            max_length=1200,\
        )

        if custom_params:
            for key, value in custom_params.items():
                setattr(args, key, value)
        seed = 24
        torch.manual_seed(seed) 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        vocab_file = os.path.join(os.path.dirname(__file__), '..', 'LLMs', 'scGPT_main', 'scgpt', 'save', 'vocab.json')
        model_config_file = os.path.join(os.path.dirname(__file__), '..', 'LLMs', 'scGPT_main', 'scgpt', 'save', 'args.json')
        model_file = os.path.join(os.path.dirname(__file__), '..', 'LLMs', 'scGPT_main', 'scgpt', 'save', 'best_model.pt')
        pad_token = "<pad>"
        special_tokens = [pad_token, "<cls>", "<eoc>"]
        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        vocab.set_default_index(vocab["<pad>"])
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        model = TransformerModel(
            ntoken=len(vocab),
            d_model=model_configs["embsize"],
            nhead=model_configs["nheads"],
            d_hid=model_configs["d_hid"],
            nlayers=model_configs["nlayers"],
            nlayers_cls=model_configs["n_layers_cls"],
            n_cls=1,
            vocab=vocab,
            dropout=model_configs["dropout"],
            pad_token=model_configs["pad_token"],
            pad_value=model_configs["pad_value"],
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=True,
            fast_transformer_backend="flash",
            pre_norm=False,
        )
        load_pretrained(model, torch.load(model_file, map_location='cuda'), verbose=False)
        model.to(device)

        class FinetuningModel(nn.Module):
            def __init__(self, input_size=512, hidden_size=512, num_classes=2):
                super(FinetuningModel, self).__init__()
                self.scGPTmodel = model
                #print(self.UCEmodel.transformer_encoder)
                # Adding Lora : QKV
                config = LoraConfig(r=8,
                                    lora_alpha=8,
                                    target_modules=args.ft_list,
                                    lora_dropout=0.05,
                                    bias="none",
                                    task_type="SEQ_CLS")  # [CAUSAL_LM,FEATURE_EXTRACTION,QUESTION_ANS,SEQ_2_SEQ_LM,SEQ_CLS,TOKEN_CLS]

                get_peft_model(self.scGPTmodel, config)

                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, num_classes)


            def forward(self, input_gene_ids, expr, src_key_padding_mask):
                embedding = self.scGPTmodel._encode(
                    input_gene_ids,
                    expr,
                    src_key_padding_mask=src_key_padding_mask
                )
                embedding = embedding[:, 0, :]
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

        FTmodel = FTmodel.to(device)

        train_data_loader, test_data_loader, vocab, model_configs= dataloader_FT(args)
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
            for batch_idx, batch in enumerate(train_data_loader):
                yield json.dumps({
                    'progress': {
                        'currentEpoch': epoch + 1,
                        'totalEpochs': args.ep_num,
                        'currentBatch': batch_idx + 1,
                        'totalBatches': len(train_data_loader)
                    }
                }).encode() + b'\n'
                input_gene_ids, expr, labels, masked_expr  = batch['gene'], batch['expr'], batch['lbl'], batch['masked_expr']
                input_gene_ids = input_gene_ids.to(device)
                expr = expr.to(device)
                labels = labels.to(device)
                masked_expr = masked_expr.to(device)
                optimizer.zero_grad()
                src_key_padding_mask = input_gene_ids.eq(
                    vocab[model_configs["pad_token"]]
                )
                pred, ebd = FTmodel(input_gene_ids, expr, src_key_padding_mask)
                loss = loss_function(pred, labels)

                loss.backward()
                optimizer.step()

                loss_sum += loss
                pred_all.extend(pred)
                lbl_all.extend(labels)

            train_metrics = {}
            pred_all_ = torch.stack(pred_all)
            lbl_all_ = torch.stack(lbl_all)
            train_metrics['loss'] = float(loss_sum/len(train_data_loader))
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
                for batch_idx, batch in enumerate(test_data_loader):
                    input_gene_ids, expr, labels, masked_expr  = batch['gene'], batch['expr'], batch['lbl'], batch['masked_expr']
                    input_gene_ids = input_gene_ids.to(device)
                    expr = expr.to(device)
                    labels = labels.to(device)
                    masked_expr = masked_expr.to(device)
                    src_key_padding_mask = input_gene_ids.eq(
                        vocab[model_configs["pad_token"]]
                    )
                    pred, ebd = FTmodel(input_gene_ids, expr, src_key_padding_mask)
                    loss = loss_function(pred, labels)
                    loss_sum += loss
                    pred_all.extend(pred)
                    lbl_all.extend(labels)
                    ebd_all.extend(ebd.to('cpu'))
                pred_all_ = torch.stack(pred_all)
                lbl_all_ = torch.stack(lbl_all)
                ebd_all_ = torch.stack(ebd_all)
                test_metrics = {}
                test_metrics['loss'] = float(loss_sum/len(test_data_loader))
                test_metrics['accuracy'] = float(Accuracy_score(pred_all_, lbl_all_))
                try:
                    test_metrics['aur'] = float(AUROC_score(pred_all_, lbl_all_))
                except:
                    test_metrics['aur'] = 0.0   
                test_metrics['precision'] = float(Precision_score(pred_all_, lbl_all_))
                test_metrics['recall'] = float(Recall_score(pred_all_, lbl_all_))
                test_metrics['f1'] = float(F1_score(pred_all_, lbl_all_))
            final_metrics['final_train'] = train_metrics    
            final_metrics['final_test'] = test_metrics
            final_metrics['epochs'].append(epoch)
            final_metrics['train_loss'].append(train_metrics['loss'])
            final_metrics['test_loss'].append(test_metrics['loss'])
            yield json.dumps({
                'metrics': final_metrics
            }).encode() + b'\n'
        os.makedirs(os.path.join(working_dir, 'scGPT_model'), exist_ok=True)
        model_save_path = os.path.join(working_dir, 'scGPT_model', 'best_model.pth')
        torch.save(model.state_dict(), model_save_path)
        yield json.dumps({
            'metrics': final_metrics,
            'model_path': model_save_path
        }).encode() + b'\n'

    except Exception as e:
        print(f"Error in scGPT finetuning: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
