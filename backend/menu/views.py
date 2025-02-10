from rest_framework import viewsets
from rest_framework.permissions import AllowAny
from menu.models import Menu

from menu.serializers import MenuSerializer

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import io
import os
from datetime import datetime
from menu.UCE_main.dataset_making import dataset_generator, shape  # Import the dataset_generator and shape functions
import shutil
from accelerate import Accelerator
from menu.UCE_main.get_ebd import main as get_ebd_main
import traceback  # 添加这个导入
import sys  # 添加这个导入
from django.http import StreamingHttpResponse, HttpResponse
import json
from .model_handlers.uce_handler import (
    process_uce_model,
    finetune_uce_model,
    train_fixed_embeddings_uce
)
from .model_handlers.tgpt_handler import (
    process_tgpt_model,
    finetune_tgpt_model,
    train_fixed_embeddings_tgpt
)


class MenuViewSet(viewsets.ModelViewSet):
    queryset = Menu.objects.all()
    serializer_class = MenuSerializer
    permission_classes = [AllowAny]

@api_view(['POST'])
def upload_csv(request):
    if 'files' not in request.FILES:
        return Response({'error': 'No files uploaded'}, 
                       status=status.HTTP_400_BAD_REQUEST)
    
    selected_model = request.POST.get('model')
    if not selected_model:
        return Response({'error': 'No model selected'}, 
                       status=status.HTTP_400_BAD_REQUEST)

    # Create timestamp for unique folder names
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_path = os.path.join(os.path.dirname(__file__), "UCE_DB")
    input_dir = os.path.join(base_path, "input", current_time)
    output_dir = os.path.join(base_path, "labels&samples", current_time)
    
    # Create input directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)
    
    # Save uploaded files
    files = request.FILES.getlist('files')
    results = []
    saved_file_paths = []
    
    for csv_file in files:
        if not csv_file.name.endswith('.csv'):
            return Response({
                'error': f'File {csv_file.name} must be CSV format'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            file_path = os.path.join(input_dir, csv_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in csv_file.chunks():
                    destination.write(chunk)
            
            saved_file_paths.append(file_path)
            df = pd.read_csv(file_path)
            results.append({
                'filename': csv_file.name,
                'rows_processed': len(df),
                'columns': list(df.columns),
                'saved_path': file_path
            })
            
        except Exception as e:
            return Response({
                'error': f'Error processing file {csv_file.name}: {str(e)}'
            }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        if selected_model == 'UCE':
            return process_uce_model(input_dir, output_dir, results)
        elif selected_model == 'tGPT':
            return process_tgpt_model(input_dir, output_dir, results)
        elif selected_model == 'scGPT':
            return process_scgpt_model(input_dir, output_dir, results)
        else:
            return Response({
                'error': f'Model {selected_model} processing not implemented yet'
            }, status=status.HTTP_501_NOT_IMPLEMENTED)
            
    except Exception as e:
        print("Full error message:", str(e), file=sys.stderr)
        print("Traceback:", file=sys.stderr)
        traceback.print_exc()
        
        return Response({
            'error': f'Error in model processing: {str(e)}',
            'files_were_saved': True,
            'input_directory': input_dir
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def process_scgpt_model(input_dir, output_dir, results):
    """Process files using scGPT model"""
    # Add scGPT specific processing logic here
    raise NotImplementedError("scGPT processing not implemented yet")

@api_view(['POST'])
def clear_cache(request):
    """Clear input and output directories"""
    try:
        input_directory = request.data.get('input_directory')
        output_directory = request.data.get('output_directory')

        if input_directory and os.path.exists(input_directory):
            shutil.rmtree(input_directory)

        if output_directory and os.path.exists(output_directory):
            shutil.rmtree(output_directory)

        return Response({
            'message': 'Cache cleared successfully'
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({
            'error': f'Error clearing cache: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def finetune_model(request):
    def generate_progress():
        try:
            # Get custom parameters from request
            custom_params = request.data
            selected_model = custom_params.get('model')
            
            if not selected_model:
                yield json.dumps({
                    'error': 'No model selected'
                }).encode() + b'\n'
                return

            # Get the latest directory from labels&samples
            base_path = os.path.join(os.path.dirname(__file__), "UCE_DB", "labels&samples")
            dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            if not dirs:
                yield json.dumps({
                    'error': 'No data directories found'
                }).encode() + b'\n'
                return
            
            latest_dir = max(dirs)
            working_dir = os.path.join(base_path, latest_dir)

            if selected_model == 'UCE':
                yield from finetune_uce_model(working_dir, custom_params)
            elif selected_model == 'tGPT':
                yield from finetune_tgpt_model(working_dir, custom_params)
            elif selected_model == 'scGPT':
                yield from finetune_scgpt_model(working_dir, custom_params)
            else:
                yield json.dumps({
                    'error': f'Finetuning not implemented for model {selected_model}'
                }).encode() + b'\n'
                
        except Exception as e:
            print("Error:", str(e))
            print("Traceback:")
            traceback.print_exc()
            yield json.dumps({
                'error': str(e)
            }).encode() + b'\n'
    
    return StreamingHttpResponse(
        generate_progress(),
        content_type='application/x-ndjson'
    )

def finetune_scgpt_model(working_dir, custom_params):
    """Finetune scGPT model"""
    yield json.dumps({
        'error': 'scGPT finetuning not implemented yet'
    }).encode() + b'\n'

@api_view(['POST'])
def train_fixed_embeddings(request):
    def generate_progress():
        try:
            custom_params = request.data
            selected_model = custom_params.get('model')
            
            if not selected_model:
                yield json.dumps({
                    'error': 'No model selected'
                }).encode() + b'\n'
                return

            base_path = os.path.join(os.path.dirname(__file__), "UCE_DB", "labels&samples")
            dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            if not dirs:
                yield json.dumps({
                    'error': 'No data directories found'
                }).encode() + b'\n'
                return
            
            latest_dir = max(dirs)
            working_dir = os.path.join(base_path, latest_dir)

            if selected_model == 'UCE':
                yield from train_fixed_embeddings_uce(working_dir, custom_params)
            elif selected_model == 'tGPT':
                yield from train_fixed_embeddings_tgpt(working_dir, custom_params)
            elif selected_model == 'scGPT':
                yield from train_fixed_embeddings_scgpt(working_dir, custom_params)
            else:
                yield json.dumps({
                    'error': f'Fixed embeddings training not implemented for model {selected_model}'
                }).encode() + b'\n'
                
        except Exception as e:
            print("Error:", str(e))
            print("Traceback:")
            traceback.print_exc()
            yield json.dumps({
                'error': str(e)
            }).encode() + b'\n'
    
    return StreamingHttpResponse(
        generate_progress(),
        content_type='application/x-ndjson'
    )

def train_fixed_embeddings_scgpt(working_dir, custom_params):
    """Train fixed embeddings for scGPT model"""
    yield json.dumps({
        'error': 'scGPT fixed embeddings training not implemented yet'
    }).encode() + b'\n'

@api_view(['POST'])
def download_model(request):
    try:
        file_path = request.data.get('file_path')
        print(f"Attempting to download file from: {file_path}")  # Debug log
        
        if not file_path:
            return Response({
                'error': 'No file path provided'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Convert relative path to absolute path if necessary
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.dirname(__file__), file_path)
            
        print(f"Absolute file path: {file_path}")  # Debug log
        
        if not os.path.exists(file_path):
            print(f"File not found at: {file_path}")  # Debug log
            return Response({
                'error': f'File not found at: {file_path}'
            }, status=status.HTTP_404_NOT_FOUND)
            
        # Get file size
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} bytes")  # Debug log
            
        with open(file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            response['Content-Length'] = file_size
            return response
            
    except Exception as e:
        print(f"Error in download_model: {str(e)}")  # Debug log
        print("Traceback:")
        traceback.print_exc()
        return Response({
            'error': f'Error downloading file: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)