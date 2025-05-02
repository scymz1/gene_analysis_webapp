#!/bin/bash

# Kill existing processes
# echo "Stopping existing services..."
# pkill -f "python manage.py runserver"
# pkill -f "npm run start"
sleep 2  # Wait for processes to stop

# Start Django backend
cd backend
source ~/miniconda3/bin/activate
conda activate shapeRNA
nohup gunicorn geneAnalysis.wsgi:application --bind 0.0.0.0:21008 --timeout 300 > django.log 2>&1 &
echo "Django backend started on port 8000"

# Build and start Next.js frontend
cd ../next-js-front
# echo "Building Next.js application..."
# npm run build
nohup npx next start -p 21007 -H 0.0.0.0 > nextjs.log 2>&1 &
echo "Next.js frontend started on port 3000" 
