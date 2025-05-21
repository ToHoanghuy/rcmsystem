@echo off
REM Run the recommendation system with MongoDB integration

echo Starting Recommendation System with MongoDB data...
cd %~dp0
python main_mongodb.py
