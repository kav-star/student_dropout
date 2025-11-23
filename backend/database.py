"""
MongoDB database operations
"""
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import pandas as pd
from datetime import datetime
from backend.config import MONGODB_URI, DATABASE_NAME, STUDENTS_COLLECTION, PREDICTIONS_COLLECTION

class DatabaseManager:
    """Handles all MongoDB operations"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(MONGODB_URI)
            # Test connection
            self.client.admin.command('ping')
            print("✅ Connected to MongoDB Atlas successfully!")
            
            self.db = self.client[DATABASE_NAME]
            self.students = self.db[STUDENTS_COLLECTION]
            self.predictions = self.db[PREDICTIONS_COLLECTION]
            
        except ConnectionFailure as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def get_all_students(self):
        """Get all student records as DataFrame"""
        try:
            students = list(self.students.find({}, {'_id': 0}))
            if not students:
                print("⚠️ No students found in database")
                return pd.DataFrame()
            return pd.DataFrame(students)
        except Exception as e:
            print(f"❌ Error fetching students: {e}")
            return pd.DataFrame()
    
    def add_student(self, student_data):
        """Add a new student record"""
        try:
            student_data['created_at'] = datetime.utcnow()
            result = self.students.insert_one(student_data)
            print(f"✅ Student added with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except DuplicateKeyError:
            print("⚠️ Student with this ID already exists")
            return None
        except Exception as e:
            print(f"❌ Error adding student: {e}")
            return None
    
    def get_student_by_id(self, student_id):
        """Get student by Student_ID"""
        try:
            student = self.students.find_one({'Student_ID': student_id}, {'_id': 0})
            return student
        except Exception as e:
            print(f"❌ Error fetching student: {e}")
            return None
    
    def update_student(self, student_id, update_data):
        """Update student information"""
        try:
            result = self.students.update_one(
                {'Student_ID': student_id},
                {'$set': update_data}
            )
            if result.modified_count > 0:
                print(f"✅ Student {student_id} updated successfully")
                return True
            else:
                print(f"⚠️ No changes made to student {student_id}")
                return False
        except Exception as e:
            print(f"❌ Error updating student: {e}")
            return False
    
    def delete_student(self, student_id):
        """Delete a student record"""
        try:
            result = self.students.delete_one({'Student_ID': student_id})
            if result.deleted_count > 0:
                print(f"✅ Student {student_id} deleted successfully")
                return True
            else:
                print(f"⚠️ Student {student_id} not found")
                return False
        except Exception as e:
            print(f"❌ Error deleting student: {e}")
            return False
    
    def save_prediction(self, student_id, student_data, prediction_result):
        """Save prediction result to database"""
        try:
            prediction_record = {
                'Student_ID': student_id,
                'student_data': student_data,
                'prediction': prediction_result['prediction'],
                'prediction_label': prediction_result['prediction_label'],
                'dropout_probability': prediction_result['dropout_probability'],
                'risk_level': prediction_result['risk_level'],
                'confidence': prediction_result['confidence'],
                'top_features': prediction_result['top_features'],
                'predicted_at': datetime.utcnow()
            }
            result = self.predictions.insert_one(prediction_record)
            print(f"✅ Prediction saved with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Error saving prediction: {e}")
            return None
    
    def get_predictions_history(self, student_id=None, limit=100):
        """Get prediction history"""
        try:
            query = {'Student_ID': student_id} if student_id else {}
            predictions = list(self.predictions.find(query, {'_id': 0}).sort('predicted_at', -1).limit(limit))
            return predictions
        except Exception as e:
            print(f"❌ Error fetching predictions: {e}")
            return []
    
    def get_statistics(self):
        """Get database statistics"""
        try:
            stats = {
                'total_students': self.students.count_documents({}),
                'total_predictions': self.predictions.count_documents({}),
                'high_risk_count': self.predictions.count_documents({'risk_level': 'High Risk'}),
                'medium_risk_count': self.predictions.count_documents({'risk_level': 'Medium Risk'}),
                'low_risk_count': self.predictions.count_documents({'risk_level': 'Low Risk'})
            }
            return stats
        except Exception as e:
            print(f"❌ Error fetching statistics: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            print("✅ MongoDB connection closed")


# Global database manager instance
db_manager = None

def get_db_manager():
    """Get or create database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager