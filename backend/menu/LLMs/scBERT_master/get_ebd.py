# -*- coding: utf-8 -*-

import argparse
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset


from performer_pytorch import PerformerLM

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=8192, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=1, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')

parser.add_argument("--data_path", type=str, default='/blue/qsong1/wang.qing/benchmark_dataset_API/scBert/samples/', help='Path of data for finetune.')

parser.add_argument("--model_path", type=str, default='./panglao_pretrain.pth', help='Path of pretrained model.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='finetune', help='Finetuned model name.')

args = parser.parse_args()


SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num+1
VALIDATE_EVERY = args.valid_every

PATIENCE = 10
UNASSIGN_THRES = 0.0

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

model_name = args.model_name
ckpt_dir = args.ckpt_dir


device = 'cuda' if torch.cuda.is_available() else 'cpu'


seed_all(SEED)



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
model = model.to(device)


files_list = os.listdir(args.data_path)
#Load data
for file_name in files_list:
    print('processing ' + file_name)
    dataset = SCDataset(args.data_path+file_name)
    loader = DataLoader(dataset, batch_size=3,shuffle=False)
    pred_all = []
    model.eval()
    pbar = tqdm(loader)
    for data in pbar:
        data = data.to(device)
        pred = model(data)
        pred_all.extend(pred.detach().cpu().numpy())


    pred_all = np.vstack(pred_all)
    np.save('/blue/qsong1/wang.qing/benchmark_dataset_API/scBert/embeds/'+file_name[:-12]+'_embeds.npy',pred_all)
    print('saved '+file_name)


