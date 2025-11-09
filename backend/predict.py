import joblib
import pandas as pd
import numpy as np
import json
import os

class DropoutPredictor:
    """Class to handle dropout predictions"""
    
    def __init__(self, model_dir=None):
        """Initialize and load all model components"""
        # Try to find model directory
        if model_dir is None:
            possible_dirs = ['model', '../model', 'model/', '../model/']
            for d in possible_dirs:
                if os.path.exists(os.path.join(d, 'random_forest_model.pkl')):
                    model_dir = d
                    break
        
        if model_dir is None:
            raise FileNotFoundError("Could not find model directory!")
        
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.metadata = None
        
        # Load all components
        self._load_model_components()
    
    def _load_model_components(self):
        """Load model, scaler, encoders, and metadata"""
        try:
            # Load the trained Random Forest model
            model_path = os.path.join(self.model_dir, 'random_forest_model.pkl')
            self.model = joblib.load(model_path)
            print("✅ Model loaded successfully")
            
            # Load the StandardScaler
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded successfully")
            
            # Load Label Encoders
            encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
            self.label_encoders = joblib.load(encoders_path)
            print("✅ Label encoders loaded successfully")
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Add valid values to metadata if not present
            if 'valid_values' not in self.metadata:
                self.metadata['valid_values'] = self._extract_valid_values()
            
            print("✅ Model metadata loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model components: {e}")
            raise
    
    def _extract_valid_values(self):
        """Extract valid values from label encoders"""
        valid_values = {}
        for col, encoder in self.label_encoders.items():
            # Convert to list and filter out NaN values
            classes = [str(c) for c in encoder.classes_ if not (isinstance(c, float) and np.isnan(c))]
            valid_values[col] = classes
        return valid_values
    
    def preprocess_input(self, student_data):
        """
        Preprocess student data before prediction
        
        Args:
            student_data (dict): Dictionary with student features
            
        Returns:
            pd.DataFrame: Preprocessed data ready for prediction
        """
        try:
            # Create DataFrame from input
            input_df = pd.DataFrame([student_data])
            
            # Get feature columns in the correct order
            feature_columns = self.metadata['feature_columns']
            categorical_columns = self.metadata['categorical_columns']
            numerical_columns = self.metadata['numerical_columns']
            
            # Ensure all required features are present
            for col in feature_columns:
                if col not in input_df.columns:
                    raise ValueError(f"Missing required feature: {col}")
            
            # Reorder columns to match training
            input_df = input_df[feature_columns]
            
            # Apply label encoding to categorical columns
            for col in categorical_columns:
                if col in input_df.columns:
                    le = self.label_encoders[col]
                    
                    # Handle missing values
                    if input_df[col].isna().any():
                        # Check if encoder has NaN in classes
                        has_nan = any(pd.isna(c) if isinstance(c, float) else str(c) == 'nan' 
                                     for c in le.classes_)
                        if not has_nan:
                            # Fill with first class if NaN not in training
                            input_df[col] = input_df[col].fillna(le.classes_[0])
                    
                    # Transform the column
                    try:
                        input_df[col] = le.transform(input_df[col])
                    except ValueError as e:
                        # Handle unseen labels
                        print(f"⚠️ Warning: Unseen value in {col}, using default")
                        input_df[col] = 0  # Default to first class
            
            # Apply standard scaling to numerical columns
            input_df[numerical_columns] = self.scaler.transform(input_df[numerical_columns])
            
            return input_df
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            raise
    
    def predict(self, student_data):
        """
        Make dropout prediction for a student
        
        Args:
            student_data (dict): Dictionary with student features
            
        Returns:
            dict: Prediction results with probabilities
        """
        try:
            # Preprocess the input
            processed_data = self.preprocess_input(student_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            probabilities = self.model.predict_proba(processed_data)[0]
            
            # Get feature importance for this prediction
            feature_importance = dict(zip(
                self.metadata['feature_columns'],
                self.model.feature_importances_
            ))
            
            # Sort by importance
            top_features = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
            
            # Prepare result
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Dropout' if prediction == 1 else 'No Dropout',
                'dropout_probability': float(probabilities[1]),
                'no_dropout_probability': float(probabilities[0]),
                'risk_level': self._get_risk_level(probabilities[1]),
                'confidence': float(max(probabilities)),
                'top_features': top_features
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            raise
    
    def _get_risk_level(self, dropout_prob):
        """Determine risk level based on dropout probability"""
        if dropout_prob >= 0.7:
            return 'High Risk'
        elif dropout_prob >= 0.4:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    def calculate_engineered_features(self, student_data):
        """
        Calculate engineered features from raw input
        
        Args:
            student_data (dict): Raw student data
            
        Returns:
            dict: Student data with engineered features added
        """
        # Age-Standard Gap
        student_data['Age_Standard_Gap'] = student_data['Age'] - student_data['Standard'] - 5
        
        # Attendance-Score Interaction
        student_data['Attendance_Score_Interaction'] = (
            student_data['Attendance'] * student_data['Previous_Score'] / 100
        )
        
        # Financial Stress
        student_data['Financial_Stress'] = int(
            (student_data['Family_Income'] == 'Low') and 
            (student_data['Scholarship'] == 'No')
        )
        
        return student_data
    
    def get_valid_values(self):
        """Get valid values for categorical features"""
        return self.metadata.get('valid_values', {})


# Create a global predictor instance
predictor = None

def get_predictor():
    """Get or create predictor instance"""
    global predictor
    if predictor is None:
        predictor = DropoutPredictor()
    return predictor


# Test function
if __name__ == "__main__":
    print("\n" + "="*50)
    print("TESTING PREDICTION MODULE")
    print("="*50)
    
    # Initialize predictor
    pred = get_predictor()
    
    # Test case 1: High risk student
    test_student_1 = {
        'Age': 16,
        'Gender': 'Male',
        'Standard': 9,
        'Caste': 'SC',
        'Area': 'Hebbal',
        'School': 'Govt Primary School',
        'Attendance': 45.0,
        'Previous_Score': 35.0,
        'Parental_Education': 'Primary',
        'Family_Income': 'Low',
        'Distance': 18.0,
        'Scholarship': 'No',
        'Special_Care': 'Minority',
        'Parent_Type': 'Single Parent',
        'Age_Standard_Gap': 2,
        'Attendance_Score_Interaction': 15.75,
        'Financial_Stress': 1
    }
    
    print("\n📊 Test Student 1 (High Risk):")
    result1 = pred.predict(test_student_1)
    print(f"   Prediction: {result1['prediction_label']}")
    print(f"   Risk Level: {result1['risk_level']}")
    print(f"   Dropout Probability: {result1['dropout_probability']:.2%}")
    print(f"   Confidence: {result1['confidence']:.2%}")
    
    # Test case 2: Low risk student
    test_student_2 = {
        'Age': 14,
        'Gender': 'Female',
        'Standard': 8,
        'Caste': 'General',
        'Area': 'Rajajinagar',
        'School': 'Govt High School',
        'Attendance': 95.0,
        'Previous_Score': 85.0,
        'Parental_Education': 'Graduate',
        'Family_Income': 'High',
        'Distance': 3.0,
        'Scholarship': 'Yes',
        'Special_Care': 'Disability',
        'Parent_Type': 'Both',
        'Age_Standard_Gap': 1,
        'Attendance_Score_Interaction': 80.75,
        'Financial_Stress': 0
    }
    
    print("\n📊 Test Student 2 (Low Risk):")
    result2 = pred.predict(test_student_2)
    print(f"   Prediction: {result2['prediction_label']}")
    print(f"   Risk Level: {result2['risk_level']}")
    print(f"   Dropout Probability: {result2['dropout_probability']:.2%}")
    print(f"   Confidence: {result2['confidence']:.2%}")
    
    print("\n" + "="*50)
    print("✅ Prediction module working correctly!")
    print("="*50)