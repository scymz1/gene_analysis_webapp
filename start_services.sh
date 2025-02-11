#!/bin/bash

# Kill existing processes
echo "Stopping existing services..."
pkill -f "python manage.py runserver"
pkill -f "npm run start"
sleep 2  # Wait for processes to stop

# Start Django backend
cd backend
source /media/volume/PhenoProfiler_volume/venv/bin/activate
nohup python manage.py runserver 0.0.0.0:8000 > django.log 2>&1 &
echo "Django backend started on port 8000"

# Build and start Next.js frontend
cd ../next-js-front
# echo "Building Next.js application..."
# npm run build
nohup npm run start > nextjs.log 2>&1 &
echo "Next.js frontend started on port 3000" 