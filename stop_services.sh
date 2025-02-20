#!/bin/bash

# netstat -tulpn | grep :3000

# Kill Django backend
pkill -f "python manage.py runserver"

# Kill Next.js frontend
pkill -f "npm run start"

echo "Services stopped" 