"""
One-time script to migrate CSV data to MongoDB
Run this ONCE after setting up MongoDB Atlas
"""
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_db_manager

def migrate_csv_to_mongodb():
    """Migrate CSV data to MongoDB"""
    print("\n" + "="*50)
    print("MIGRATING CSV DATA TO MONGODB")
    print("="*50)
    
    # Load CSV
    # csv_path = 'data/school_dropout_data_with_features.csv'
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} records from CSV")
    except FileNotFoundError:
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    # Connect to MongoDB
    db = get_db_manager()
    
    # Clear existing data (optional - comment out if you want to keep old data)
    print("\n⚠️ Clearing existing student records...")
    db.students.delete_many({})
    
    # Insert students
    print(f"\n📤 Uploading {len(df)} students to MongoDB...")
    students = df.to_dict('records')
    
    try:
        result = db.students.insert_many(students)
        print(f"✅ Successfully migrated {len(result.inserted_ids)} students!")
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        return
    
    # Verify
    count = db.students.count_documents({})
    print(f"\n✅ Verification: {count} students in MongoDB")
    
    # Show sample
    print("\n📊 Sample record:")
    sample = db.students.find_one({}, {'_id': 0})
    for key, value in list(sample.items())[:5]:
        print(f"   {key}: {value}")
    
    print("\n" + "="*50)
    print("✅ MIGRATION COMPLETE!")
    print("="*50)
    
    db.close()

if __name__ == "__main__":
    migrate_csv_to_mongodb()