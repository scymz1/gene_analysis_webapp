from .benchmarking_dataloader_FT import dataloader
import argparse
from tqdm import tqdm
from torch import optim
from .tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score
import pandas as pd
from torch import nn
from peft import LoraConfig, get_peft_model
import pickle
import warnings
import torch
from .utils import figshare_download
from .modelFT import TransformerModel
import os
import json




warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description='AI4Bio')

# training parameters
parser.add_argument('--ep_num', type=int, default=3, help='epoch number of training')
parser.add_argument('--train_batch_size', type=int, default=80, help='')
parser.add_argument('--test_batch_size', type=int, default=80, help='')
parser.add_argument('--label_path', type=str, default='/blue/qsong1/wang.qing/benchmark_dataset_API/UCE/labels/', help='')
parser.add_argument('--data_path', type=str, default="/blue/qsong1/wang.qing/benchmark_dataset_API/UCE/samples/", help='')
parser.add_argument('--ft_list', type=list, default=['out_proj'], help='')
parser.add_argument('--train_rate', type=float, default=0.8, help='')

parser.add_argument('--lr', type=int, default=0.0001, help='')
parser.add_argument('--pe_idx_path', type=str, default='./10k_pbmcs_proc_pe_idx.torch', help='')
parser.add_argument('--chroms_path', type=str, default='./10k_pbmcs_proc_chroms.pkl', help='')
parser.add_argument('--starts_path', type=str, default='./10k_pbmcs_proc_starts.pkl', help='')
parser.add_argument('--pad_length', type=int, default=1536,help='Batch size.')
parser.add_argument('--sample_size', type=int, default=1024,help='Number of genes sampled for cell sentence')
parser.add_argument("--cls_token_idx", type=int, default=3,help="CLS token index")
parser.add_argument("--CHROM_TOKEN_OFFSET", type=int, default=143574,help="Offset index, tokens after this mark are chromosome identifiers")
parser.add_argument("--chrom_token_left_idx", type=int, default=1,help="Chrom token left index")
parser.add_argument("--chrom_token_right_idx", type=int, default=2,help="Chrom token right index")
parser.add_argument("--pad_token_idx", type=int, default=0,help="PAD token index")
parser.add_argument('--token_dim', type=int, default=5120,help='Token dimension.')
parser.add_argument('--d_hid', type=int, default=5120,help='Hidden dimension.')
parser.add_argument('--nlayers', type=int, default=4,help='Number of transformer layers.')
parser.add_argument('--output_dim', type=int, default=1280,help='Output dimension.')
parser.add_argument('--model_loc', type=str,default=None,help='Location of the model.')
parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output files')



def get_model(args):
    #### Set up the model ####
    token_dim = args.token_dim
    emsize = 1280  # embedding dimension
    d_hid = args.d_hid  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = args.nlayers  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 20  # number of heads in nn.MultiheadAttention
    dropout = 0.05  # dropout probability
    model = TransformerModel(token_dim=token_dim, d_model=emsize, nhead=nhead,
                             d_hid=d_hid,
                             nlayers=nlayers, dropout=dropout,
                             output_dim=args.output_dim)
    if args.model_loc is None:
        args.model_loc = "./model_files/4layer_model.torch"
        figshare_download("https://figshare.com/ndownloader/files/42706576",args.model_loc)
    # intialize as empty
    empty_pe = torch.zeros(145469, 5120)
    empty_pe.requires_grad = False
    model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
    model.load_state_dict(torch.load(args.model_loc, map_location="cpu"),strict=True)
    #print(model)


    class FinetuningModel(nn.Module):
        def __init__(self, input_size=1280, hidden_size=512, num_classes=2):
            super(FinetuningModel, self).__init__()
            self.UCEmodel = model
            #print(self.UCEmodel.transformer_encoder)
            # Adding Lora : QKV
            config = LoraConfig(r=8,
                                lora_alpha=8,
                                target_modules=args.ft_list,
                                lora_dropout=0.05,
                                bias="none",
                                task_type="SEQ_CLS")  # [CAUSAL_LM,FEATURE_EXTRACTION,QUESTION_ANS,SEQ_2_SEQ_LM,SEQ_CLS,TOKEN_CLS]

            get_peft_model(self.UCEmodel.transformer_encoder, config)  #self.transformer_encoder_lora =
            # self.UCEmodel.transformer_encoder = self.transformer_encoder_lora
            # print(self.UCEmodel.transformer_encoder)
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)


        def forward(self, seq, mask):
            seq = self.UCEmodel.pe_embedding(seq.long())
            seq = nn.functional.normalize(seq,dim=2)
            _, embedding = self.UCEmodel.forward(seq, mask=mask)  # 200*1280
            x = self.fc1(embedding)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x, embedding

    FTmodel = FinetuningModel(input_size=1280, hidden_size=512, num_classes=2)
    total_params = sum(p.numel() for p in FTmodel.parameters())
    print(f"Total number of parameters: {total_params}")

    return FTmodel

def run(args=None):
    """Run the finetuning process with args"""
    if args is None:
        args = parser.parse_args([])
        
    seed = 24
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(args)
    loss_function = torch.nn.BCEWithLogitsLoss()
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(model_parameters, lr=args.lr)

    model = model.to(device)

    metrics = {
        'epochs': [],
        'train_loss': [],
        'test_loss': [],
        'train_metrics': [],
        'test_metrics': [],
        'final_train': None,
        'final_test': None
    }

    train_data_loader, test_data_loader = dataloader(args)
    total_batches = len(train_data_loader)
    
    for epoch in range(args.ep_num):
        loss_sum = 0
        pred_all = []
        lbl_all = []
        ebd_all = []
    
        # Training loop
        model.train()
        for batch_idx, b in enumerate(train_data_loader):
            yield {
                'progress': {
                    'currentEpoch': epoch + 1,
                    'totalEpochs': args.ep_num,
                    'currentBatch': batch_idx + 1,
                    'totalBatches': total_batches
                }
            }

            batch_sentences, mask, cell_sentences, labels = b[0], b[1], b[2], b[3]
            batch_sentences = batch_sentences.permute(1, 0).long()
            pred, ebd = model(batch_sentences.to(device), mask=mask.to(device))
            loss = loss_function(pred, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss_sum = loss_sum + loss
            # pred_all.extend(pred)
            # lbl_all.extend(labels)
            # ebd_all.extend(ebd.to('cpu'))
            loss_sum += loss.item()
            pred_all.extend(pred.detach().cpu())
            lbl_all.extend(labels.cpu())
            ebd_all.extend(ebd.detach().cpu())

        # Calculate training metrics
        pred_all_ = torch.stack(pred_all)
        lbl_all_ = torch.stack(lbl_all)
        ebd_all_ = torch.stack(ebd_all)
        
        # Apply sigmoid to predictions since we're using BCEWithLogitsLoss
        pred_probs = torch.sigmoid(pred_all_)
        pred_labels = (pred_probs > 0.5).float()
        
        train_metrics = {
            'accuracy': float(Accuracy_score(pred_labels, lbl_all_)),
            'precision': float(Precision_score(pred_labels, lbl_all_)),
            'recall': float(Recall_score(pred_labels, lbl_all_)),
            'f1': float(F1_score(pred_labels, lbl_all_))
        }
        
        print(f"Train metrics: {train_metrics}")  # Debug print
        
        metrics['train_metrics'].append(train_metrics)
        metrics['train_loss'].append(float(loss_sum/len(train_data_loader)))
        metrics['epochs'].append(epoch + 1)

        if epoch == args.ep_num - 1:
            metrics['final_train'] = train_metrics
            
            # Calculate final test metrics
            model.eval()
            with torch.no_grad():
                test_pred_all = []
                test_lbl_all = []
                test_loss_sum = 0
                
                for b in test_data_loader:
                    batch_sentences, mask, cell_sentences, labels = b[0], b[1], b[2], b[3]
                    batch_sentences = batch_sentences.permute(1, 0).long()
                    pred, _ = model(batch_sentences.to(device), mask=mask.to(device))
                    loss = loss_function(pred, labels.to(device))
                    
                    test_loss_sum += loss.item()
                    test_pred_all.extend(pred.cpu())
                    test_lbl_all.extend(labels.cpu())
                
                test_pred_all_ = torch.stack(test_pred_all)
                test_lbl_all_ = torch.stack(test_lbl_all)
                
                # Apply sigmoid to test predictions
                test_pred_probs = torch.sigmoid(test_pred_all_)
                test_pred_labels = (test_pred_probs > 0.5).float()
                
                test_metrics = {
                    'accuracy': float(Accuracy_score(test_pred_labels, test_lbl_all_)),
                    'precision': float(Precision_score(test_pred_labels, test_lbl_all_)),
                    'recall': float(Recall_score(test_pred_labels, test_lbl_all_)),
                    'f1': float(F1_score(test_pred_labels, test_lbl_all_))
                }
                
                print(f"Test metrics: {test_metrics}")  # Debug print
                
                metrics['final_test'] = test_metrics
                metrics['test_loss'].append(float(test_loss_sum/len(test_data_loader)))

    # Save results
    save_dir = os.path.join(args.output_dir, 'finetuned_model')
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'model.pth')
    records_path = os.path.join(save_dir, 'testset_records.pkl')
    
    torch.save(model.state_dict(), model_path)
    
    records = {
        'pred': test_pred_labels,
        'label': test_lbl_all_,
        'ebd': ebd_all_
    }
    with open(records_path, 'wb') as f:
        pickle.dump(records, f)

    yield {
        'metrics': metrics,
        'model_path': model_path,
        'test_records_path': records_path
    }

# Only parse args if running as main script
if __name__ == '__main__':
    args = parser.parse_args()
    run(args)


