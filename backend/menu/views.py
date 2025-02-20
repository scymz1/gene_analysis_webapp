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
from datetime import datetime, timedelta
from menu.UCE_main.dataset_making import dataset_generator, shape  # Import the dataset_generator and shape functions
import shutil
from accelerate import Accelerator
from menu.UCE_main.get_ebd import main as get_ebd_main
import traceback  # 添加这个导入
import sys  # 添加这个导入
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse, FileResponse
from django.views.decorators.http import require_http_methods
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
from .model_handlers.scGPT_handler import (
    process_scGPT_model,
    train_fixed_embeddings_scGPT,
    finetune_scGPT
)

from .model_handlers.scFoundation_handler import (
    process_scfoundation_model, 
    train_fixed_embeddings_scFoundation,
    finetune_scFoundation
)
from .model_handlers.scBERT_handler import (
    process_scBERT_model,
    train_fixed_embeddings_scBERT,
    finetune_scBERT 
)
from .model_handlers.OpenBioMed_handler import (
    process_OpenBioMed_model,
    train_fixed_embeddings_OpenBioMed,
    finetune_OpenBioMed
)
from .model_handlers.CellPLM_handler import (
    process_CellPLM_model,
    train_fixed_embeddings_CellPLM,
    finetune_CellPLM
)
from .model_handlers.GeneFormer_handler import (
    process_GeneFormer_model,
    train_fixed_embeddings_GeneFormer,
    finetune_GeneFormer
)
import zipfile
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
@require_http_methods(["POST"])
def bulk_download(request):
    try:
        data = json.loads(request.body)
        if 'path' not in request.GET:
            base_path = '/media/volume/Minghao_webserver/dataset/lxndt_filter'
        else:
            base_path = os.path.join('/media/volume/Minghao_webserver/dataset', request.GET.get('path'))
        files = data.get('files', [])
        print(f"Base path: {base_path}")
        print(f"Files: {files}")
        
        # Create a zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_name in files:
                file_path = os.path.join(base_path, file_name)
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        # Add file to zip with its name as archive name
                        zip_file.write(file_path, file_name)
                    elif os.path.isdir(file_path):
                        # Walk through directory and add all files
                        for root, dirs, dir_files in os.walk(file_path):
                            for f in dir_files:
                                full_path = os.path.join(root, f)
                                # Calculate relative path for the archive name
                                archive_name = os.path.join(
                                    file_name,  # Keep the original folder name
                                    os.path.relpath(full_path, file_path)  # Add relative path within the folder
                                )
                                zip_file.write(full_path, archive_name)
        
        # Prepare response
        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename=selected_files.zip'
        return response
        
    except Exception as e:
        print(f"Error in bulk_download: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return HttpResponse(str(e), status=500)

@require_http_methods(["GET"])
def download_file(request):
    if 'path' not in request.GET:
        file_path = '/media/volume/Minghao_webserver/dataset/lxndt_filter'
    else:
        file_path = os.path.join('/media/volume/Minghao_webserver/dataset', request.GET.get('path'))
    
    if not file_path or not os.path.exists(file_path):
        return HttpResponse(status=404)
    
    try:
        return FileResponse(
            open(file_path, 'rb'),
            as_attachment=True,
            filename=os.path.basename(file_path)
        )
    except Exception as e:
        return HttpResponse(str(e), status=500)


def list_files(request):
    if 'path' not in request.GET:
        path = '/media/volume/Minghao_webserver/dataset/lxndt_filter'
    else:
        path = os.path.join('/media/volume/Minghao_webserver/dataset', request.GET.get('path'))
    
    try:
        files = []
        with os.scandir(path) as entries:
            for entry in entries:
                stats = entry.stat()
                files.append({
                    'name': entry.name,
                    'isDirectory': entry.is_dir(),
                    'size': stats.st_size,
                    'modifiedTime': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                })
        
        return JsonResponse({
            'files': sorted(files, key=lambda x: (not x['isDirectory'], x['name'].lower()))
        }, safe=False, content_type="application/json")
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

class MenuViewSet(viewsets.ModelViewSet):
    queryset = Menu.objects.all()
    serializer_class = MenuSerializer
    permission_classes = [AllowAny]

def cleanup_old_folders(base_path, hours=12):
    """Delete folders older than specified hours"""
    try:
        current_time = datetime.now()
        # Check both input and labels&samples directories
        for subdir in ['input', 'labels&samples']:
            dir_path = os.path.join(base_path, subdir)
            if not os.path.exists(dir_path):
                continue
                
            for folder_name in os.listdir(dir_path):
                try:
                    # Parse the timestamp from folder name
                    folder_time = datetime.strptime(folder_name, '%Y%m%d_%H%M%S')
                    # Check if folder is older than specified hours
                    if current_time - folder_time > timedelta(hours=hours):
                        folder_path = os.path.join(dir_path, folder_name)
                        if os.path.exists(folder_path):
                            shutil.rmtree(folder_path)
                            print(f"Deleted old folder: {folder_path}")
                except ValueError:
                    # Skip folders that don't match timestamp format
                    continue
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

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
    
    # Clean up old folders before creating new ones
    cleanup_old_folders(base_path)
    
    # Create input directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)
    
    # Save uploaded files
    files = request.FILES.getlist('files')
    results = []
    # saved_file_paths = []
    
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
            
            # saved_file_paths.append(file_path)
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
            return process_scGPT_model(input_dir, output_dir, results)
        elif selected_model == 'scFoundation':
            return process_scfoundation_model(input_dir, output_dir, results)
        elif selected_model == 'scBERT':
            return process_scBERT_model(input_dir, output_dir, results)
        elif selected_model == 'Openbiomed(cellLM)':
            return process_OpenBioMed_model(input_dir, output_dir, results) 
        elif selected_model == 'CellPLM':
            return process_CellPLM_model(input_dir, output_dir, results)
        elif selected_model == 'GeneFormer':
            return process_GeneFormer_model(input_dir, output_dir, results)
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


# def process_scgpt_model(input_dir, output_dir, results):
#     """Process files using scGPT model"""
#     # Add scGPT specific processing logic here
#     raise NotImplementedError("scGPT processing not implemented yet")

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
                yield from finetune_scGPT(working_dir, custom_params)
            elif selected_model == 'scFoundation':
                yield from finetune_scFoundation(working_dir, custom_params)
            elif selected_model == 'scBERT':
                yield from finetune_scBERT(working_dir, custom_params)
            elif selected_model == 'Openbiomed(cellLM)':
                yield from finetune_OpenBioMed(working_dir, custom_params)
            elif selected_model == 'CellPLM':
                yield from finetune_CellPLM(working_dir, custom_params)
            elif selected_model == 'GeneFormer':
                yield from finetune_GeneFormer(working_dir, custom_params)
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


            
            if not custom_params.get('output_directory'):
                base_path = os.path.join(os.path.dirname(__file__), "UCE_DB", "labels&samples")
                dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                if not dirs:
                    yield json.dumps({
                        'error': 'No data directories found'
                    }).encode() + b'\n'
                    return
                latest_dir = max(dirs)
                working_dir = os.path.join(base_path, latest_dir)
            else:
                print(f"Using provided output directory: {custom_params.get('output_directory')}")
                working_dir = custom_params.get('output_directory')
            
            

            if selected_model == 'UCE':
                yield from train_fixed_embeddings_uce(working_dir, custom_params)
            elif selected_model == 'tGPT':
                yield from train_fixed_embeddings_tgpt(working_dir, custom_params)
            elif selected_model == 'scGPT':
                yield from train_fixed_embeddings_scGPT(working_dir, custom_params)
            elif selected_model == 'scFoundation':
                yield from train_fixed_embeddings_scFoundation(working_dir, custom_params)
            elif selected_model == 'scBERT':
                yield from train_fixed_embeddings_scBERT(working_dir, custom_params)
            elif selected_model == 'Openbiomed(cellLM)':
                yield from train_fixed_embeddings_OpenBioMed(working_dir, custom_params)
            elif selected_model == 'CellPLM':
                yield from train_fixed_embeddings_CellPLM(working_dir, custom_params)
            elif selected_model == 'GeneFormer':
                yield from train_fixed_embeddings_GeneFormer(working_dir, custom_params)
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