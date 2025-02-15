import json
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import scipy.sparse
from scipy.sparse import issparse
import scanpy as sc
import pickle
from rest_framework.response import Response
from rest_framework import status
import traceback
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from torch import nn
from ..LLMs.scFoundation_main.model.load import load_model_frommmf, gatherData
from ..LLMs.scFoundation_main.model.benchmarking_dataloader_EBD import dataloader
from ..LLMs.scFoundation_main.model.benchmarking_dataloader_FT import dataloader as dataloader_FT
from ..LLMs.scFoundation_main.model.tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score
from peft import LoraConfig, get_peft_model
def main_gene_selection(X_df, gene_list):
    """
    Rebuild the input data to select target genes encode protein 
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))),
                            columns=to_fill_columns,
                            index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1),
                        index=X_df.index,
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]

    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns, var

def process_scfoundation_model(input_dir, output_dir, results):
    """Process files using scFoundation model"""
    try:
        # Set random seed
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load gene list
        gene_list_path = os.path.join(os.path.dirname(__file__), "..", "LLMs", "scFoundation_main", "model", "OS_scRNA_gene_index.19264.tsv")
        gene_list_df = pd.read_csv(gene_list_path, header=0, delimiter='\t')
        gene_list = list(gene_list_df['gene_name'])

        # Load model
        ckpt_path = os.path.join(os.path.dirname(__file__), "..", "LLMs", "scFoundation_main", "model", "models", "models.ckpt")
        key = 'cell'
        pretrainmodel, pretrainconfig = load_model_frommmf(ckpt_path, key)
        pretrainmodel.eval()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectories
        samples_dir = os.path.join(output_dir, "samples")
        labels_dir = os.path.join(output_dir, "labels")
        embeds_dir = os.path.join(output_dir, "embeds")
        
        for directory in [samples_dir, labels_dir, embeds_dir]:
            os.makedirs(directory, exist_ok=True)

        # Process each file
        for result in results:
            file_path = result['saved_path']
            filename = os.path.basename(file_path)
            base_filename = os.path.splitext(filename)[0]

            # Read data
            gexpr_feature = pd.read_csv(file_path, sep='\t', index_col=0,
                                      usecols=lambda column: column != 'Condition')
            lbl = list(pd.read_csv(file_path, sep='\t', usecols=['Condition'])['Condition'])
            label = [1 if element == 'sensitive' else 0 for element in lbl]

            # Convert gene features if necessary
            if gexpr_feature.shape[1] != 19264:
                gexpr_feature, to_fill_columns, var = main_gene_selection(gexpr_feature, gene_list)
                assert gexpr_feature.shape[1] >= 19264

            # First process: get x, x_padding, pos_id
            data_dict = {}
            x_all = []
            x_padding_all = []
            pos_id_all = []

            # Process each cell
            for i in tqdm(range(gexpr_feature.shape[0]), desc="Processing initial data"):
                with torch.no_grad():
                    tmpdata = (np.log1p(gexpr_feature.iloc[i, :] / (gexpr_feature.iloc[i, :].sum()) * 1e4)).tolist()
                    totalcount = gexpr_feature.iloc[i, :].sum()
                    pretrain_gene_x = torch.tensor(tmpdata + [4.0, np.log10(totalcount)], dtype=torch.float32).unsqueeze(0).cuda(0)
                    data_gene_ids = torch.arange(19266, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)

                value_labels = pretrain_gene_x > 0
                x, x_padding = gatherData(pretrain_gene_x, value_labels, pretrainconfig['pad_token_id'])
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pretrainconfig['pad_token_id'])
                x_all.extend(x)
                x_padding_all.extend(x_padding)
                pos_id_all.extend(position_gene_ids)

            # Second process: get embeddings
            geneexpemb = []
            for i in tqdm(range(gexpr_feature.shape[0]), desc="Getting embeddings"):
                with torch.no_grad():
                    tmpdata = (np.log1p(gexpr_feature.iloc[i, :] / (gexpr_feature.iloc[i, :].sum()) * 1e4)).tolist()
                    totalcount = gexpr_feature.iloc[i, :].sum()
                    pretrain_gene_x = torch.tensor(tmpdata + [4.0, np.log10(totalcount)], dtype=torch.float32).unsqueeze(0).cuda(0)
                    data_gene_ids = torch.arange(19266, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)

                value_labels = pretrain_gene_x > 0
                x, x_padding = gatherData(pretrain_gene_x, value_labels, pretrainconfig['pad_token_id'])
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pretrainconfig['pad_token_id'])
                
                x = pretrainmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
                position_emb = pretrainmodel.pos_emb(position_gene_ids)
                x += position_emb
                geneemb = pretrainmodel.encoder(x, x_padding)
                
                geneemb1 = geneemb[:, -1, :]
                geneemb2 = geneemb[:, -2, :]
                geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
                geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
                geneembmerge = torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], axis=1)
                geneexpemb.append(geneembmerge.detach().cpu().numpy())

            geneexpemb = np.squeeze(np.array(geneexpemb))

            # Save samples (x, x_padding, pos_id)
            samples_dict = {
                'x': x_all,
                'x_padding': x_padding_all,
                'pos_id': pos_id_all
            }
            samples_file = os.path.join(samples_dir, f"{base_filename}_samples.pkl")
            with open(samples_file, "wb") as p:
                pickle.dump(samples_dict, p)

            # Save labels
            labels_file = os.path.join(labels_dir, f"{base_filename}_labels.npy")
            np.save(labels_file, np.array(label))

            # Save embeddings
            embeds_file = os.path.join(embeds_dir, f"{base_filename}_embeds.npy")
            np.save(embeds_file, geneexpemb)

            # Update result with all paths
            result['processed_paths'] = {
                'samples': samples_file,
                'labels': labels_file,
                'embeds': embeds_file
            }

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

class MLP_Classifier(nn.Module):
    def __init__(self, input_size=3072, hidden_size=512, num_classes=2):
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

def train_fixed_embeddings_scFoundation(working_dir, custom_params):
    """Train classifier using fixed embeddings from scFoundation model"""
    try:
        # Set random seed for reproducibility
        seed = 24
        torch.manual_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create args object similar to argparse
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        # Set default parameters
        args = Args(
            ep_num=10,
            train_batch_size=128,
            test_batch_size=256,
            data_path=os.path.join(working_dir, 'embeds'),
            label_path=os.path.join(working_dir, 'labels'),
            lr=0.0001,
            train_rate=0.8,
            ft_list=['none']
        )

        # Update with custom parameters if provided
        if custom_params:
            for key, value in custom_params.items():
                setattr(args, key, value)

        # Initialize model, loss function and optimizer
        model = MLP_Classifier().to(device)
        loss_function = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            [param for param in model.parameters() if param.requires_grad], 
            lr=args.lr
        )

        # Get data loaders
        train_loader, test_loader = dataloader(args)

        # Training loop
        total_batches = len(train_loader)
        final_metrics = {
            'train_loss': [],
            'test_loss': [],
            'epochs': []
        }
        for epoch in range(args.ep_num):
            # Send epoch progress
            yield json.dumps({
                'progress': {
                    'currentEpoch': epoch + 1,
                    'totalEpochs': args.ep_num,
                    'currentBatch': 0,
                    'totalBatches': total_batches
                }
            }).encode() + b'\n'

            # Training phase
            model.train()
            train_loss = 0
            train_pred_all = []
            train_lbl_all = []

            for batch_idx, (embeds_batch, labels_batch) in enumerate(train_loader):
                # Send batch progress
                yield json.dumps({
                    'progress': {
                        'currentEpoch': epoch + 1,
                        'totalEpochs': args.ep_num,
                        'currentBatch': batch_idx + 1,
                        'totalBatches': total_batches
                    }
                }).encode() + b'\n'

                embeds_batch = embeds_batch.float().to(device)
                labels_batch = labels_batch.to(device)

                pred = model(embeds_batch)
                loss = loss_function(pred, labels_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_pred_all.extend(pred.detach().cpu())
                train_lbl_all.extend(labels_batch.cpu())

            # Calculate training metrics
            train_metrics = calculate_metrics(train_pred_all, train_lbl_all)
            train_metrics['loss'] = train_loss / len(train_loader)

            # Testing phase
            model.eval()
            test_loss = 0
            test_pred_all = []
            test_lbl_all = []

            with torch.no_grad():
                for batch_idx, (embeds_batch, labels_batch) in enumerate(test_loader):
                    embeds_batch = embeds_batch.float().to(device)
                    labels_batch = labels_batch.to(device)

                    pred = model(embeds_batch)
                    loss = loss_function(pred, labels_batch)

                    test_loss += loss.item()
                    test_pred_all.extend(pred.cpu())
                    test_lbl_all.extend(labels_batch.cpu())

            # Calculate test metrics
            test_metrics = calculate_metrics(test_pred_all, test_lbl_all)
            test_metrics['loss'] = test_loss / len(test_loader)

            final_metrics['final_train'] = train_metrics
            final_metrics['final_test'] = test_metrics
            final_metrics['train_loss'].append(train_metrics['loss'])
            final_metrics['test_loss'].append(test_metrics['loss'])
            final_metrics['epochs'].append(epoch + 1)

            # Send current metrics
            yield json.dumps({
                'metrics': final_metrics
            }).encode() + b'\n'

        # Save model
        model_save_path = os.path.join(working_dir, 'best_model.pth')
        torch.save(model.state_dict(), model_save_path)

        # Send final results
        yield json.dumps({
            'metrics': final_metrics,
            'model_path': model_save_path
        }).encode() + b'\n'

    except Exception as e:
        error_msg = f"scFoundation fixed embeddings training error: {str(e)}\nFull traceback:\n{traceback.format_exc()}"
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

def calculate_metrics(pred_all, lbl_all):
    """Helper function to calculate metrics"""
    pred_all_ = torch.stack(pred_all)
    lbl_all_ = torch.stack(lbl_all)
    
    # pred_probs = torch.sigmoid(pred_all)
    # pred_labels = torch.round(pred_probs)
    
    return {
        'accuracy': Accuracy_score(pred_all_, lbl_all_),
        'f1': F1_score(pred_all_, lbl_all_),
        'auc': AUROC_score(pred_all_, lbl_all_),
        'precision': Precision_score(pred_all_, lbl_all_),
        'recall': Recall_score(pred_all_, lbl_all_)
    } 


def get_finetune_model(args):
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "LLMs", "scFoundation_main", "model", "models", "models.ckpt")
    key = 'cell'
    model, _ = load_model_frommmf(ckpt_path, key)
    # print(model)

    class FinetuningModel(nn.Module):
        def __init__(self, input_size=307, hidden_size=512, num_classes=2, args=None):
            super(FinetuningModel, self).__init__()
            self.scFoundationmodel = model
            #print(self.UCEmodel.transformer_encoder)
            # Adding Lora : QKV
            config = LoraConfig(r=8,
                                lora_alpha=8,
                                target_modules=args.ft_list,
                                lora_dropout=0.05,
                                bias="none",
                                task_type="SEQ_CLS")  # [CAUSAL_LM,FEATURE_EXTRACTION,QUESTION_ANS,SEQ_2_SEQ_LM,SEQ_CLS,TOKEN_CLS]

            get_peft_model(self.scFoundationmodel, config)  #self.transformer_encoder_lora =
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)


        def forward(self, x, x_padding, pos_id):
            x = self.scFoundationmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)  # exp+pos
            position_emb = self.scFoundationmodel.pos_emb(pos_id)
            x += position_emb
            geneemb = self.scFoundationmodel.encoder(x, x_padding)

            geneemb1 = geneemb[:, -1, :]
            geneemb2 = geneemb[:, -2, :]
            geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
            geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
            embedding = torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], axis=1)

            x = self.fc1(embedding)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x, embedding

    FTmodel = FinetuningModel(input_size=3072, hidden_size=512, num_classes=2, args=args)
    total_params = sum(p.numel() for p in FTmodel.parameters())
    print(f"Total number of parameters: {total_params}")

    return FTmodel

def finetune_scFoundation(working_dir, custom_params):
    """Train classifier using fixed embeddings from scFoundation model"""
    try:

        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        # Set default parameters
        args = Args(
            ep_num=3,
            train_batch_size=1,
            test_batch_size=1,
            data_path=os.path.join(working_dir, 'samples'),
            label_path=os.path.join(working_dir, 'labels'),
            lr=0.0001,
            train_rate=0.8,
            ft_list=['out_proj'],
            pad_length=1536,
            sample_size=1024
        )

        # Update with custom parameters if provided
        if custom_params:
            for key, value in custom_params.items():
                setattr(args, key, value)

        # Set random seed for reproducibility
        seed = 24
        torch.manual_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize model, loss function and optimizer
        model = get_finetune_model(args=args)
        loss_function = torch.nn.BCEWithLogitsLoss()
        model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = optim.AdamW(model_parameters, lr=args.lr)

        model = model.to(device)
        # Get data loaders
        train_loader, test_loader = dataloader_FT(args)
        total_batches = len(train_loader)
        final_metrics = {
            'train_loss': [],
            'test_loss': [],
            'epochs': []
        }
        for epoch in range(args.ep_num):
            yield json.dumps({
                'progress': {
                    'currentEpoch': epoch + 1,
                    'totalEpochs': args.ep_num,
                    'currentBatch': 0,
                    'totalBatches': total_batches
                }
            }).encode() + b'\n'

            # Training phase
            model.train()
            loss_sum = 0
            pred_all = []
            lbl_all = []

            for batch_idx, batches in enumerate(train_loader):
                yield json.dumps({
                    'progress': {
                        'currentEpoch': epoch + 1,
                        'totalEpochs': args.ep_num,
                        'currentBatch': batch_idx + 1,
                        'totalBatches': total_batches
                    }
                }).encode() + b'\n'
                x, x_padding, pos_id, labels  = batches[0], batches[1], batches[2], batches[3]
                pred, ebd = model(x, x_padding, pos_id)
                loss = loss_function(pred, labels.to(device))


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum = loss_sum + loss
                pred_all.extend(pred)
                lbl_all.extend(labels)
            train_metrics = {}
            pred_all_ = torch.stack(pred_all)
            lbl_all_ = torch.stack(lbl_all)
            train_metrics['accuracy'] = float(Accuracy_score(pred_all_, lbl_all_))
            train_metrics['f1'] = float(F1_score(pred_all_, lbl_all_))
            try:
                train_metrics['auc'] = float(AUROC_score(pred_all_, lbl_all_))
            except:
                train_metrics['auc'] = 0.0
            train_metrics['precision'] = float(Precision_score(pred_all_, lbl_all_))
            train_metrics['recall'] = float(Recall_score(pred_all_, lbl_all_))
            train_metrics['loss'] = float(loss_sum / len(batches))

            loss_sum = 0
            pred_all = []
            lbl_all = []
            ebd_all = []
            with torch.no_grad():
                for batch_idx, batches in enumerate(test_loader):
                    x, x_padding, pos_id, labels = batches[0], batches[1], batches[2], batches[3]
                    pred, ebd = model(x, x_padding, pos_id)
                    loss = loss_function(pred, labels.to(device))

                    loss_sum = loss_sum + loss
                    pred_all.extend(pred.to('cpu'))
                    lbl_all.extend(labels.to('cpu'))
                    ebd_all.extend(ebd.to('cpu'))
                test_metrics = {}
                pred_all_ = torch.stack(pred_all)
                lbl_all_ = torch.stack(lbl_all)
                ebd_all_ = torch.stack(ebd_all)
                test_metrics['accuracy'] = float(Accuracy_score(pred_all_, lbl_all_))
                test_metrics['f1'] = float(F1_score(pred_all_, lbl_all_))
                try:
                    test_metrics['auc'] = float(AUROC_score(pred_all_, lbl_all_))
                except:
                    test_metrics['auc'] = 0.0
                test_metrics['precision'] = float(Precision_score(pred_all_, lbl_all_))
                test_metrics['recall'] = float(Recall_score(pred_all_, lbl_all_))
                test_metrics['loss'] = float(loss_sum / len(batches))
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

            # if epoch == args.ep_num-1:
            #     roc={}
            #     roc['pred']=pred_all_
            #     roc['label']=lbl_all_
            #     roc['ebd']=ebd_all_
            #     with open('./output/testset_records.pkl', 'wb') as rocpkl:
            #         pickle.dump(roc, rocpkl)
            #     torch.save(model.state_dict(), './output/model.pth')

    except Exception as e:
        error_msg = f"scFoundation fixed embeddings training error: {str(e)}\nFull traceback:\n{traceback.format_exc()}"
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