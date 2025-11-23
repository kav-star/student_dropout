"""
Configuration file for MongoDB connection
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'dropout_prediction')

# Collections
STUDENTS_COLLECTION = 'students'
PREDICTIONS_COLLECTION = 'predictions'
MODEL_METRICS_COLLECTION = 'model_metrics'

# Validate configuration
if not MONGODB_URI:
    raise ValueError("MONGODB_URI not found in .env file!")