import scanpy as sc
import pandas as pd
import pickle
import numpy as np

seg_label = np.load('./annotation_data/data/Segerstolpe-test-label.npy')
seg_name = np.load('./annotation_data/data/Segerstolpe-str_label.npy')


emb_path="./annotation_data/data/seg-emb.pkl"
emb=None
label=None
f=open(emb_path, 'rb')
while 1:
    try:
        sub_pkl = pickle.load(f)
        tmp_emb = sub_pkl["emb"]
        tmp_label = sub_pkl["label"]
        if emb is None:
            emb = tmp_emb
            label = tmp_label
        else:
            emb = np.vstack([emb,tmp_emb])
            label = np.concatenate([label,tmp_label])
    except:
        break

emb = emb[-seg_label.shape[0]:,:]
label = label[-seg_label.shape[0]:]


emb_path="./annotation_data/data/seg-cellemb.pkl"
cellemb=None
label=None
f=open(emb_path, 'rb')
while 1:
    try:
        sub_pkl = pickle.load(f)
        tmp_emb = sub_pkl["emb"]
        tmp_label = sub_pkl["label"]
        if cellemb is None:
            cellemb = tmp_emb
            label = tmp_label
        else:
            cellemb = np.vstack([cellemb,tmp_emb])
            label = np.concatenate([label,tmp_label])
    except:
        break

cellemb = cellemb[-seg_label.shape[0]:,:]
label = label[-seg_label.shape[0]:]

for i in range(seg_label.shape[0]):
    assert seg_label[i]==label[i]


y_pred = np.argmax(emb,1)