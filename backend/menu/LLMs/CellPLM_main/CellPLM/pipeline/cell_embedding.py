import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
import anndata as ad
from tqdm import tqdm
from copy import deepcopy
from ..utils.eval import downstream_eval, aggregate_eval_results
from ..utils.data import XDict, TranscriptomicDataset
from typing import List, Literal, Union
from torch.utils.data import DataLoader
import warnings
from . import Pipeline, load_pretrain
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

class CellEmbeddingPipeline(Pipeline):
    def __init__(self,
                 pretrain_prefix: str,
                 pretrain_directory: str = './ckpt',
                 ):
        super().__init__(pretrain_prefix, {'head_type': 'embedder'}, pretrain_directory)
        self.label_encoders = None

    def fit(self, adata: ad.AnnData,
            train_config: dict = None,
            split_field: str = None,  # A field in adata.obs for representing train-test split
            train_split: str = None,  # A specific split where labels can be utilized for training
            valid_split: str = None,  # A specific split where labels can be utilized for validation
            covariate_fields: List[str] = None,  # A list of fields in adata.obs that contain cellular covariates
            label_fields: List[str] = None,  # A list of fields in adata.obs that contain cell labels
            batch_gene_list: dict = None,  # A dictionary that contains batch and gene list pairs
            ensembl_auto_conversion: bool = True,
            # A bool value indicating whether the function automativally convert symbols to ensembl id
            device: Union[str, torch.device] = 'cpu'
            ):
        raise NotImplementedError('Currently CellPLM only supports zero shot embedding instead of fine-tuning.')

    def predict(self, adata,
                inference_config: dict = None,
                covariate_fields: List[str] = None,
                batch_gene_list: dict = None,
                ensembl_auto_conversion: bool = True,
                device: Union[str, torch.device] = 'cpu'
                ):

        return self._inference(adata, device, ensembl_auto_conversion)

    def _inference(self, adata,
                device: Union[str, torch.device] = 'cpu',
                ensembl_auto_conversion: bool = True):
        self.model.to(device)
        # we change the adata to our data structure (by QW), only need x_seq
        # adata={'seq','gene_list'}

        with torch.no_grad():
            self.model.eval()
            new = {}
            new['x_seq'] = adata['x_seq'].to(device)
            x_dict = new
            out_dict, _ = self.model(x_dict, adata['gene_list'])
            pred = out_dict['pred']#[input_dict['order_list']])
            return pred

    def score(self, adata: ad.AnnData,
              evaluation_config: dict = None,
              split_field: str = None,
              target_split: str = 'test',
              covariate_fields: List[str] = None,
              label_fields: List[str] = None,
              batch_gene_list: dict = None,
              ensembl_auto_conversion: bool = True,
              device: Union[str, torch.device] = 'cpu'
              ):
        if evaluation_config and 'batch_size' in evaluation_config:
            batch_size = evaluation_config['batch_size']
        else:
            batch_size = 0
        if len(label_fields) != 1:
            raise NotImplementedError(
                f'`label_fields` containing multiple labels (f{len(label_fields)}) is not implemented for evaluation of cell embedding pipeline. Please raise an issue on Github for further support.')
        if split_field:
            warnings.warn('`split_field` argument is ignored in CellEmbeddingPipeline.')
        if target_split:
            warnings.warn('`target_split` argument is ignored in CellEmbeddingPipeline.')
        if covariate_fields:
            warnings.warn('`covariate_fields` argument is ignored in CellEmbeddingPipeline.')
        if batch_gene_list:
            warnings.warn('`batch_gene_list` argument is ignored in CellEmbeddingPipeline.')

        adata = adata.copy()
        pred = self._inference(adata, batch_size, device)
        adata.obsm['emb'] = pred.cpu().numpy()
        if 'method' in evaluation_config and evaluation_config['method'] == 'rapids':
            sc.pp.neighbors(adata, use_rep='emb', method='rapids')
        else:
            sc.pp.neighbors(adata, use_rep='emb')
        best_ari = -1
        best_nmi = -1
        for res in range(1, 15, 1):
            res = res / 10
            if 'method' in evaluation_config and evaluation_config['method'] == 'rapids':
                import rapids_singlecell as rsc
                rsc.tl.leiden(adata, resolution=res, key_added='leiden')
            else:
                sc.tl.leiden(adata, resolution=res, key_added='leiden')
            ari_score = adjusted_rand_score(adata.obs['leiden'].to_numpy(), adata.obs[label_fields[0]].to_numpy())
            if ari_score > best_ari:
                best_ari = ari_score
            nmi_score = normalized_mutual_info_score(adata.obs['leiden'].to_numpy(), adata.obs[label_fields[0]].to_numpy())
            if nmi_score > best_nmi:
                best_nmi = nmi_score
        return {'ari': best_ari, 'nmi': best_nmi}




