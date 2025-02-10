import os
import json
import traceback
from accelerate import Accelerator
from ..UCE_main.dataset_making import dataset_generator, shape
from ..UCE_main.get_ebd import main as get_ebd_main
from rest_framework.response import Response
from rest_framework import status

def process_uce_model(input_dir, output_dir, results):
    """Process files using UCE model"""
    try:
        # Process the saved files using dataset_generator
        dataset_dirs = dataset_generator(
            directory_path=input_dir,
            output_dir=output_dir,
            is_sorted=True,
            seq_length=8192
        )
        
        # Generate shape dictionaries
        shape_files = shape(
            directory_path_samples=dataset_dirs['samples_dir'],
            directory_path_labels=dataset_dirs['labels_dir'],
            output_dir=output_dir
        )
        
        # Generate embeddings
        embeddings_dir = generate_embeddings(
            samples_dir=dataset_dirs['samples_dir'],
            shape_file_path=shape_files['samples_shape_path'],
            output_dir=output_dir
        )

        return Response({
            'message': 'Files processed successfully with UCE model',
            'files_processed': len(results),
            'input_directory': os.path.relpath(input_dir),
            'output_directory': os.path.relpath(output_dir),
            'samples_shape_file': os.path.relpath(shape_files['samples_shape_path']),
            'labels_shape_file': os.path.relpath(shape_files['labels_shape_path']),
            'embeddings_directory': os.path.relpath(embeddings_dir),
            'details': results
        }, status=status.HTTP_200_OK)
    except Exception as e:
        raise Exception(f"UCE processing error: {str(e)}")

def generate_embeddings(samples_dir, shape_file_path, output_dir):
    """Generate embeddings using the get_ebd model"""
    class Args:
        def __init__(self):
            self.dir_data = samples_dir
            self.dir_shape = shape_file_path
            self.dir_ebds = os.path.join(output_dir, 'embeddings')
            self.dir = output_dir
            
            # Update base path to use absolute path from handler location
            base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "UCE_main", "model_files")
            self.model_loc = os.path.join(base_path, "4layer_model.torch")
            self.spec_chrom_csv_path = os.path.join(base_path, "species_chrom.csv")
            self.token_file = os.path.join(base_path, "all_tokens.torch")
            self.protein_embeddings_dir = os.path.join(base_path, "protein_embeddings")
            self.offset_pkl_path = os.path.join(base_path, "species_offsets.pkl")
            
            # Other default parameters
            self.batch_size = 25
            self.pad_length = 1536
            self.pad_token_idx = 0
            self.chrom_token_left_idx = 1
            self.chrom_token_right_idx = 2
            self.cls_token_idx = 3
            self.CHROM_TOKEN_OFFSET = 143574
            self.sample_size = 1024
            self.CXG = True
            self.nlayers = 4
            self.output_dim = 1280
            self.d_hid = 5120
            self.token_dim = 5120
            self.multi_gpu = False
            self.adata_path = None
            self.species = "human"
            self.filter = True
            self.skip = True

    # Create embeddings directory
    os.makedirs(os.path.join(output_dir, 'embeddings'), exist_ok=True)

    # Run get_ebd
    args = Args()
    accelerator = Accelerator(project_dir=args.dir)
    get_ebd_main(args, accelerator)
    
    return os.path.join(output_dir, 'embeddings')

def finetune_uce_model(working_dir, custom_params):
    """Finetune UCE model"""
    try:
        from ..UCE_main.benchmarking_main_FT import parser, run as run_finetuning
        
        args = parser.parse_args([])
        
        # Update args with custom parameters
        if 'ep_num' in custom_params:
            args.ep_num = min(max(int(custom_params['ep_num']), 0), 10)
        if 'train_rate' in custom_params:
            args.train_rate = min(max(float(custom_params['train_rate']), 0), 1)
        if 'lr' in custom_params:
            args.lr = max(float(custom_params['lr']), 0)
        
        # Set paths
        args.output_dir = working_dir
        args.label_path = os.path.join(working_dir, 'labels')
        args.data_path = os.path.join(working_dir, 'samples')
        
        # Set model file paths
        temp_path = os.path.dirname(os.path.dirname(__file__))
        base_path = os.path.join(temp_path, "UCE_main/model_files")
        args.model_loc = os.path.join(base_path, "4layer_model.torch")
        args.spec_chrom_csv_path = os.path.join(base_path, "species_chrom.csv")
        args.token_file = os.path.join(base_path, "all_tokens.torch")
        args.protein_embeddings_dir = os.path.join(base_path, "protein_embeddings")
        args.offset_pkl_path = os.path.join(base_path, "species_offsets.pkl")
        args.pe_idx_path = os.path.join(temp_path, "UCE_main", "10k_pbmcs_proc_pe_idx.torch")
        args.chroms_path = os.path.join(temp_path, "UCE_main", "10k_pbmcs_proc_chroms.pkl")
        args.starts_path = os.path.join(temp_path, "UCE_main", "10k_pbmcs_proc_starts.pkl")
        
        # Set other parameters
        args.train_batch_size = 8
        args.test_batch_size = 10
        args.ft_list = ['out_proj']
        args.pad_length = 1536
        args.sample_size = 1024
        args.cls_token_idx = 3
        args.CHROM_TOKEN_OFFSET = 143574
        args.chrom_token_left_idx = 1
        args.chrom_token_right_idx = 2
        args.pad_token_idx = 0
        args.token_dim = 5120
        args.d_hid = 5120
        args.nlayers = 4
        args.output_dim = 1280
        args.CXG = True
        args.multi_gpu = False
        args.adata_path = None
        args.species = "human"
        args.filter = True
        args.skip = True
        
        # Check required files
        required_files = [
            args.model_loc,
            args.spec_chrom_csv_path,
            args.token_file,
            args.offset_pkl_path,
            args.pe_idx_path,
            args.protein_embeddings_dir
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        for progress_update in run_finetuning(args):
            if isinstance(progress_update, dict):
                yield json.dumps(progress_update).encode() + b'\n'
                
    except Exception as e:
        error_msg = f"UCE finetuning error: {str(e)}\nFull traceback:\n{traceback.format_exc()}"
        yield json.dumps({
            'error': error_msg
        }).encode() + b'\n'

def train_fixed_embeddings_uce(working_dir, custom_params):
    """Train fixed embeddings for UCE model"""
    try:
        from ..UCE_main.benchmarking_main_EBD import parser, run as run_ebd_training
        
        args = parser.parse_args([])
        
        # Set paths
        args.output_dir = working_dir
        args.label_path = os.path.join(working_dir, 'labels')
        args.data_path = os.path.join(working_dir, 'embeddings')
        
        # Set fixed parameters
        args.train_rate = 0.8
        args.train_batch_size = 8
        args.test_batch_size = 10
        args.ep_num = 3
        args.lr = 0.0001
        
        # Update with custom parameters
        if custom_params:
            args.ep_num = int(custom_params.get('ep_num', args.ep_num))
            args.train_rate = float(custom_params.get('train_rate', args.train_rate))
            args.lr = float(custom_params.get('lr', args.lr))
        
        for progress_update in run_ebd_training(args):
            if isinstance(progress_update, dict):
                yield json.dumps(progress_update).encode() + b'\n'
                
    except Exception as e:
        yield json.dumps({
            'error': f'UCE fixed embeddings training error: {str(e)}'
        }).encode() + b'\n' 