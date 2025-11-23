# 🎓 DropoutAI - Student Dropout Prediction System

A data-driven software solution developed to analyse student dropout patterns across government schools in Karnataka. This project supports the Right to Education initiative by enabling focused policy interventions based on insights from dropout data. *Dropout AI* AI-powered student dropout prediction system using Random Forest machine learning to help educational institutions identify at-risk students and take proactive measures.

## 🌟 Features

- **Real-time Predictions**: Instant dropout risk assessment for individual students
- **Cloud Database**: Secure data storage using MongoDB Atlas
- **Interactive Dashboard**: Comprehensive data visualizations and insights
- **Analytics**: Explore patterns and correlations in student data
- **Risk Categorization**: High, Medium, Low risk levels with actionable recommendations
- **Model Performance**: Random Forest classifier with high accuracy

## 🚀 Demo

![Dashboard Preview](screenshots/dashboard.png)

## 📊 Tech Stack

### **Frontend**
- **Streamlit** - Interactive web application framework

### **Backend**
- **Python 3.8+** - Core programming language
- **scikit-learn** - Machine learning models
- **Random Forest Classifier** - Primary prediction model

### **Database**
- **MongoDB Atlas** - Cloud-based NoSQL database
- **pymongo** - MongoDB driver for Python

### **Data Processing & Visualization**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Plotly** - Interactive visualizations

## 🛠️ Installation

### **Prerequisites**

- Python 3.8 or higher
- MongoDB Atlas account (Free tier available)
- pip (Python package manager)

## 🎯 Usage

### 1. Dashboard
View comprehensive insights about student data including:
- Student demographics
- Attendance and performance distributions
- Model performance metrics

### 2. Predict Risk
Enter student information to get:
- Dropout probability
- Risk level (High/Medium/Low)
- Top risk factors
- Personalized recommendations

### 3. Analytics
Explore patterns and correlations:
- Attendance vs Performance
- Distance vs Attendance
- Family income distribution
- Parental education levels

## 📈 Model Performance

- **Accuracy**: 78.95% 
- **AUC Score**: 0.833 
- **Algorithm**: Random Forest Classifier

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Project Link: [https://github.com/kav-star/student_dropout](https://github.com/kav-star/student_dropout)

## 🙏 Acknowledgments

- CBD_7 - Capstone Project Team
- Presidency University

Guide:
- Dr. Abdul Majid

Team Members:
- Tanushree R
- Kavya J
- Kavya S

## Step 4: Setup Instructions

1. **Clone the repository**
```bash
   git clone https://github.com/kav-star/student_dropout.git
   cd student_dropout
```

2. **Create virtual environment**
```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
```

3. **Upgrade pip (Important!)**
```bash
python -m pip install --upgrade pip
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```
5. **Setup MongoDB Atlas**
- Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
- Sign up for a free account
- Create a new cluster (FREE M0 tier)
- Go to **Database Access** → Add New Database User
- Create username and password (save these!)
- Grant **Read and Write** permissions
- Go to **Network Access** → Add IP Address
- For development: Allow access from anywhere (`0.0.0.0/0`)
- For production: Add your specific IP addresses
- Go to **Database** → **Connect** → **Connect your application**
- Copy the connection string
- Replace `<password>` with your actual password

6. **Configure Environment Variables**
Create a `.env` file in the root directory:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and add your MongoDB credentials:

```env
MONGODB_URI=mongodb+srv://YOUR_USERNAME:YOUR_PASSWORD@YOUR_CLUSTER.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=dropout_prediction
```

7. **Migrate Data to MongoDB**
Run the migration script to upload CSV data to MongoDB Atlas:

```bash
python data/migrate_to_mongo.py
```

8. **Run the application**
```bash
streamlit run app.py
```
The app will automatically open at `http://localhost:8501`