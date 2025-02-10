import torch
from .benchmarking_dataloader_EBD import dataloader
import argparse
from tqdm import tqdm
from torch import nn
from torch import optim
from .tool import Accuracy_score, F1_score, AUROC_score, Recall_score, Precision_score
import pandas as pd
import os
import csv
import pickle

# Create parser but don't parse args immediately
parser = argparse.ArgumentParser(description='AI4Bio')

# training parameters
parser.add_argument('--ep_num', type=int, default=10, help='epoch number of training')
parser.add_argument('--train_batch_size', type=int, default=128, help='')
parser.add_argument('--test_batch_size', type=int, default=256, help='')
parser.add_argument('--label_path', type=str, default='/blue/qsong1/wang.qing/benchmark_dataset_API/UCE/labels/', help='')
parser.add_argument('--data_path', type=str, default="/blue/qsong1/wang.qing/benchmark_dataset_API/UCE/ebds/", help='')
parser.add_argument('--lr', type=float, default=0.0001, help='')  # Changed from int to float
parser.add_argument('--train_rate', type=float, default=0.8, help='')
parser.add_argument('--ft_list', type=list, default=['none'], help='')
parser.add_argument('--output_dir', type=str, default='./output', help='')

class MLP_Classifier(nn.Module):
    def __init__(self, input_size=1280, hidden_size=512, num_classes=2):
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

def run(args=None):
    """Run the fixed embeddings training process with given args"""
    if args is None:
        # Only parse args if none are provided
        args = parser.parse_args([])  # Pass empty list to avoid reading sys.argv
        
    seed = 24
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MLP_Classifier().to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    train_data_loader, test_data_loader = dataloader(args)
    total_batches = len(train_data_loader)

    # Setup metrics dictionary
    metrics = {
        'epochs': [],
        'train_loss': [],
        'test_loss': [],
        'train_metrics': [],
        'test_metrics': [],
        'final_train': None,
        'final_test': None
    }

    for epoch in range(args.ep_num):
        loss_sum = 0
        pred_all = []
        lbl_all = []
    
        # Training loop
        model.train()
        for batch_idx, b in enumerate(train_data_loader):
            # Yield progress update
            yield {
                'progress': {
                    'currentEpoch': epoch + 1,
                    'totalEpochs': args.ep_num,
                    'currentBatch': batch_idx + 1,
                    'totalBatches': total_batches
                }
            }
            # print(f"Progress: Epoch {epoch + 1}/{args.ep_num}, Batch {batch_idx + 1}/{total_batches}")

            embeddings, labels = b[0], b[1]
            pred = model(embeddings.to(device))
            loss = loss_function(pred, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum = loss_sum + loss
            pred_all.extend(pred)
            lbl_all.extend(labels)

        # Calculate training metrics
        pred_all_ = torch.stack(pred_all)
        lbl_all_ = torch.stack(lbl_all)
        
        train_metrics = {
            'accuracy': float(Accuracy_score(pred_all_, lbl_all_)),
            'precision': float(Precision_score(pred_all_, lbl_all_)),
            'recall': float(Recall_score(pred_all_, lbl_all_)),
            'f1': float(F1_score(pred_all_, lbl_all_))
        }
        
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
                    embeddings, labels = b[0], b[1]
                    pred = model(embeddings.to(device))
                    loss = loss_function(pred, labels.to(device))
                    
                    test_loss_sum += loss
                    test_pred_all.extend(pred.to('cpu'))
                    test_lbl_all.extend(labels)
                
                test_pred_all_ = torch.stack(test_pred_all)
                test_lbl_all_ = torch.stack(test_lbl_all)
                
                metrics['final_test'] = {
                    'accuracy': float(Accuracy_score(test_pred_all_, test_lbl_all_)),
                    'precision': float(Precision_score(test_pred_all_, test_lbl_all_)),
                    'recall': float(Recall_score(test_pred_all_, test_lbl_all_)),
                    'f1': float(F1_score(test_pred_all_, test_lbl_all_))
                }
                metrics['test_loss'].append(float(test_loss_sum/len(test_data_loader)))

    # Save model and return metrics
    model_path = os.path.join(args.output_dir, 'fixed_embeddings_model.pth')
    torch.save(model.state_dict(), model_path)
    
    # Yield final metrics
    yield {
        'metrics': metrics,
        'model_path': model_path
    }

if __name__ == '__main__':
    run()


