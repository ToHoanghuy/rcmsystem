#!/bin/bash
# Run the recommendation system with MongoDB integration

echo "Starting Recommendation System with MongoDB data..."
cd "$(dirname "$0")"
python main_mongodb.py
