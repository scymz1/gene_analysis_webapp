import os
import logging
from django.conf import settings
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
import numpy as np
import torch
from skimage.transform import resize
from PIL import Image
import io
from .PhenoProfilerModel.models import PhenoProfiler

# Set up logging
logger = logging.getLogger(__name__)

# Create your views here.

class MorphologyProfileView(APIView):
    def post(self, request):
        try:
            logger.info("Received analysis request")
            print("Files in request:", request.FILES)

            # Get the uploaded files
            files = []
            for i in range(5):
                file_key = f'image_{i}'
                if file_key not in request.FILES:
                    return Response(
                        {
                            'error': f'Missing image {i+1}',
                            'detail': f'Please upload exactly 5 channel images. Image {i+1} is missing.'
                        }, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
                files.append(request.FILES[file_key])
            
            logger.info(f"Received {len(files)} files")

            # Process images
            images = []
            for i, file in enumerate(files):
                try:
                    # Read image from InMemoryUploadedFile
                    image = Image.open(io.BytesIO(file.read()))
                    
                    # Check if image is grayscale
                    if image.mode not in ['L', 'I']:
                        return Response(
                            {
                                'error': f'Invalid image format for image {i+1}',
                                'detail': f'Image {i+1} must be grayscale. Current mode: {image.mode}'
                            },
                            status=status.HTTP_400_BAD_REQUEST
                        )
                    
                    # Convert to numpy array and resize
                    image_array = np.array(image)
                    resized_image = resize(image_array, (448, 448), anti_aliasing=True)
                    images.append(resized_image)
                    
                except Exception as e:
                    logger.error(f"Error processing image {i+1}: {str(e)}")
                    return Response(
                        {
                            'error': f'Error processing image {i+1}',
                            'detail': str(e)
                        },
                        status=status.HTTP_400_BAD_REQUEST
                    )

            # Stack images
            try:
                images = np.stack(images)
                images_tensor = torch.tensor(images).float().cuda()
            except Exception as e:
                logger.error(f"Error stacking images: {str(e)}")
                return Response(
                    {
                        'error': 'Error preparing images',
                        'detail': str(e)
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Load model
            try:
                model = PhenoProfiler().cuda()
                model_path = os.path.join(settings.BASE_DIR, 'phenoProfiler', 'PhenoProfilerModel', 'best.pt')
                logger.info(f"Loading model from: {model_path}")
                
                if not os.path.exists(model_path):
                    return Response(
                        {
                            'error': 'Model file not found',
                            'detail': f'Could not find model file at {model_path}'
                        },
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                
                model.load_state_dict(torch.load(model_path, weights_only=True))
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                return Response(
                    {
                        'error': 'Error loading model',
                        'detail': str(e)
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Generate embeddings
            try:
                with torch.no_grad():
                    image_features = model.image_encoder(images_tensor.unsqueeze(0))
                    image_embeddings = model.image_projection(image_features)
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                return Response(
                    {
                        'error': 'Error generating embeddings',
                        'detail': str(e)
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Save embeddings
            try:
                embeddings = image_embeddings.cpu().numpy()
                output = io.BytesIO()
                np.save(output, embeddings)
                output.seek(0)
            except Exception as e:
                logger.error(f"Error saving embeddings: {str(e)}")
                return Response(
                    {
                        'error': 'Error saving results',
                        'detail': str(e)
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Return the file
            logger.info("Successfully processed images and generated embeddings")
            response = HttpResponse(output.read(), content_type='application/octet-stream')
            response['Content-Disposition'] = 'attachment; filename=morphology_profiles.npy'
            return response

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return Response(
                {
                    'error': 'Unexpected error',
                    'detail': str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
