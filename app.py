import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from predict import get_predictor

# Page configuration
st.set_page_config(
    page_title="DropoutAI - Student Dropout Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: #1f1f1f !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #0e1117 !important;
        font-size: 2rem !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #31333F !important;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: #1f1f1f !important;
    }
    .prediction-box h2, .prediction-box h3, .prediction-box p {
        color: #1f1f1f !important;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .medium-risk {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize predictor
@st.cache_resource
def load_predictor():
    """Load the ML model predictor"""
    return get_predictor()

@st.cache_data
def load_dataset():
    """Load the dataset"""
    possible_paths = [
        'data/school_dropout_data_with_features.csv',
        '../data/school_dropout_data_with_features.csv',
        'school_dropout_data_with_features.csv'
    ]
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            print(f"✅ Dataset loaded from: {path}")
            print(f"Columns: {list(df.columns)}")
            return df
        except:
            continue
    st.error("❌ Could not find dataset file!")
    return pd.DataFrame()

@st.cache_data
def load_model_metrics():
    """Load model metrics"""
    possible_paths = [
        'model/model_metrics.json',
        '../model/model_metrics.json'
    ]
    for path in possible_paths:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            continue
    return {}

@st.cache_data
def load_feature_importance():
    """Load feature importance"""
    possible_paths = [
        'data/feature_importance.csv',
        '../data/feature_importance.csv'
    ]
    for path in possible_paths:
        try:
            return pd.read_csv(path)
        except:
            continue
    return pd.DataFrame({'feature': [], 'importance': []})

# Sidebar navigation
def sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/student-center.png", width=80)
        st.title("DropoutAI")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🎯 Predict Risk", "📊 Analytics", "ℹ️ About", "📧 Contact"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 🎓 About")
        st.info("AI-powered student dropout prediction system using Random Forest ML model.")
        
        return page

# Dashboard Page
def dashboard_page():
    st.title("🏠 School Dropout Analysis Dashboard")
    st.markdown("### Comprehensive insights and trends for educational outcomes")
    
    # Load data
    df = load_dataset()
    metrics = load_model_metrics()
    
    if df.empty:
        st.error("No data available. Please check your data file.")
        return
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Students",
            value=f"{len(df):,}",
            delta="In Dataset"
        )
    
    with col2:
        avg_attendance = df['Attendance'].mean()
        st.metric(
            label="Avg Attendance",
            value=f"{avg_attendance:.1f}%",
            delta="Overall"
        )
    
    with col3:
        if metrics:
            st.metric(
                label="Model Accuracy",
                value=f"{metrics['test_accuracy']*100:.2f}%",
                delta="Test Set"
            )
        else:
            st.metric(label="Model Accuracy", value="N/A")
    
    with col4:
        if metrics:
            st.metric(
                label="Model AUC Score",
                value=f"{metrics['test_auc']:.3f}",
                delta="High Performance"
            )
        else:
            st.metric(label="Model AUC", value="N/A")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Student Distribution by Gender")
        gender_dist = df['Gender'].value_counts().reset_index()
        gender_dist.columns = ['Gender', 'Count']
        fig = px.bar(
            gender_dist,
            x='Gender',
            y='Count',
            color='Gender',
            color_discrete_map={'Male': '#2196F3', 'Female': '#E91E63'},
            title="Gender Distribution"
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Students by Area")
        area_dist = df['Area'].value_counts().reset_index()
        area_dist.columns = ['Area', 'Count']
        fig = px.bar(
            area_dist,
            x='Area',
            y='Count',
            color='Count',
            color_continuous_scale='Viridis',
            title="Area Distribution"
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Attendance Distribution")
        fig = px.histogram(
            df,
            x='Attendance',
            nbins=30,
            color_discrete_sequence=['#4CAF50'],
            title="Attendance Distribution (%)"
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📚 Previous Score Distribution")
        fig = px.histogram(
            df,
            x='Previous_Score',
            nbins=30,
            color_discrete_sequence=['#2196F3'],
            title="Academic Performance Distribution"
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 3
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏫 Students by School Type")
        school_dist = df['School'].value_counts().reset_index()
        school_dist.columns = ['School', 'Count']
        fig = px.pie(
            school_dist,
            values='Count',
            names='School',
            title="School Distribution"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("👨‍👩‍👧 Parent Type Distribution")
        parent_dist = df['Parent_Type'].value_counts().reset_index()
        parent_dist.columns = ['Parent_Type', 'Count']
        fig = px.pie(
            parent_dist,
            values='Count',
            names='Parent_Type',
            title="Parent Type Distribution"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Performance Section
    if metrics:
        st.markdown("---")
        st.subheader("🎯 Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        report = metrics.get('classification_report', {})
        dropout_stats = report.get('Dropout', {})
        
        with col1:
            st.metric("Precision (Dropout)", f"{dropout_stats.get('precision', 0):.3f}")
            st.metric("Recall (Dropout)", f"{dropout_stats.get('recall', 0):.3f}")
        
        with col2:
            st.metric("F1-Score (Dropout)", f"{dropout_stats.get('f1-score', 0):.3f}")
            st.metric("Support (Dropout)", f"{int(dropout_stats.get('support', 0))}")
        
        with col3:
            # Confusion Matrix
            cm = np.array(metrics.get('confusion_matrix', [[0,0],[0,0]]))
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['No Dropout', 'Dropout'],
                y=['No Dropout', 'Dropout'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16}
            ))
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)

# Predict Risk Page
def predict_page():
    st.title("🎯 Student Dropout Risk Prediction")
    st.markdown("### Enter student information to predict dropout risk")
    
    predictor = load_predictor()
    valid_values = predictor.get_valid_values()
    
    # Create two columns for input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Personal Information")
        age = st.number_input("Age", min_value=5, max_value=20, value=14, step=1)
        gender = st.selectbox("Gender", valid_values.get('Gender', ['Male', 'Female']))
        standard = st.number_input("Standard/Grade", min_value=1, max_value=12, value=8, step=1)
        caste = st.selectbox("Caste", valid_values.get('Caste', ['General', 'OBC', 'SC', 'ST']))
        
        st.subheader("🏫 Academic Information")
        attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0, 0.5)
        previous_score = st.slider("Previous Score", 0.0, 100.0, 65.0, 0.5)
    
    with col2:
        st.subheader("🏡 Family & Location")
        area = st.selectbox("Area", valid_values.get('Area', ['Rajajinagar', 'Hebbal', 'Malleswaram']))
        school = st.selectbox("School", valid_values.get('School', ['Govt Primary School', 'Govt High School']))
        parental_education = st.selectbox("Parental Education", 
                                         valid_values.get('Parental_Education', ['None', 'Primary', 'Graduate', 'Higher']))
        family_income = st.selectbox("Family Income", valid_values.get('Family_Income', ['Low', 'Medium', 'High']))
        
        st.subheader("📍 Additional Information")
        distance = st.number_input("Distance from School (km)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        scholarship = st.selectbox("Scholarship", valid_values.get('Scholarship', ['Yes', 'No']))
        special_care = st.selectbox("Special Care", 
                                    valid_values.get('Special_Care', ['None', 'Disability', 'Minority']))
        parent_type = st.selectbox("Parent Type", valid_values.get('Parent_Type', ['Both', 'Single Parent', 'Orphan']))
    
    # Predict button
    st.markdown("---")
    if st.button("🎯 Predict Dropout Risk", type="primary", use_container_width=True):
        # Prepare student data
        student_data = {
            'Age': age,
            'Gender': gender,
            'Standard': standard,
            'Caste': caste,
            'Area': area,
            'School': school,
            'Attendance': attendance,
            'Previous_Score': previous_score,
            'Parental_Education': parental_education,
            'Family_Income': family_income,
            'Distance': distance,
            'Scholarship': scholarship,
            'Special_Care': special_care,
            'Parent_Type': parent_type
        }
        
        # Calculate engineered features
        student_data = predictor.calculate_engineered_features(student_data)
        
        # Make prediction
        with st.spinner("Analyzing student data..."):
            result = predictor.predict(student_data)
        
        # Display results
        st.markdown("---")
        st.subheader("📋 Prediction Results")
        
        # Determine risk styling
        risk_level = result['risk_level']
        if 'High' in risk_level:
            risk_class = 'high-risk'
            risk_icon = '🔴'
        elif 'Medium' in risk_level:
            risk_class = 'medium-risk'
            risk_icon = '🟡'
        else:
            risk_class = 'low-risk'
            risk_icon = '🟢'
        
        # Prediction box
        st.markdown(f"""
        <div class="prediction-box {risk_class}">
            <h2>{risk_icon} {result['prediction_label']}</h2>
            <h3>Risk Level: {result['risk_level']}</h3>
            <p style="font-size: 18px;">Dropout Probability: <b>{result['dropout_probability']:.1%}</b></p>
            <p style="font-size: 18px;">Confidence: <b>{result['confidence']:.1%}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['dropout_probability'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Dropout Risk Score"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance for this prediction
            st.subheader("🔑 Top Risk Factors")
            top_features = result['top_features']
            
            feature_df = pd.DataFrame({
                'Feature': list(top_features.keys()),
                'Importance': list(top_features.values())
            })
            
            fig = px.bar(
                feature_df,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if result['dropout_probability'] > 0.7:
            st.error("""
            **Immediate Intervention Required:**
            - Schedule urgent parent-teacher meeting
            - Provide academic counseling and support
            - Consider financial assistance programs
            - Monitor attendance closely
            - Assign mentor or peer support
            """)
        elif result['dropout_probability'] > 0.4:
            st.warning("""
            **Preventive Measures Recommended:**
            - Regular monitoring of academic performance
            - Encourage participation in extracurricular activities
            - Provide tutoring if needed
            - Maintain communication with parents
            """)
        else:
            st.success("""
            **Student Doing Well:**
            - Continue current support level
            - Encourage academic excellence
            - Recognize achievements
            """)

# Analytics Page
def analytics_page():
    st.title("📊 Data Analytics & Insights")
    st.markdown("### Explore patterns and correlations in student data")
    
    df = load_dataset()
    
    if df.empty:
        st.error("No data available for analytics.")
        return
    
    # Statistical Summary
    st.subheader("📈 Statistical Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Age", f"{df['Age'].mean():.1f} years")
    with col2:
        st.metric("Avg Attendance", f"{df['Attendance'].mean():.1f}%")
    with col3:
        st.metric("Avg Score", f"{df['Previous_Score'].mean():.1f}")
    with col4:
        st.metric("Avg Distance", f"{df['Distance'].mean():.1f} km")
    
    st.markdown("---")
    
    # Correlation Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔗 Attendance vs Previous Score")
        fig = px.scatter(
            df,
            x='Attendance',
            y='Previous_Score',
            color='Gender',
            size='Age',
            hover_data=['School', 'Area'],
            title="Relationship between Attendance and Academic Performance"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📏 Distance vs Attendance")
        fig = px.scatter(
            df,
            x='Distance',
            y='Attendance',
            color='Area',
            size='Age',
            hover_data=['School'],
            title="Impact of Distance on Attendance"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # More Analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Family Income Distribution")
        income_dist = df['Family_Income'].value_counts().reset_index()
        income_dist.columns = ['Income', 'Count']
        fig = px.bar(
            income_dist,
            x='Income',
            y='Count',
            color='Income',
            title="Family Income Categories"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎓 Parental Education")
        edu_dist = df['Parental_Education'].value_counts().reset_index()
        edu_dist.columns = ['Education', 'Count']
        fig = px.bar(
            edu_dist,
            x='Education',
            y='Count',
            color='Education',
            title="Parental Education Levels"
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# About Page
def about_page():
    st.title("ℹ️ About DropoutAI")
    
    st.markdown("""
    ## 🎓 Project Overview
    
    School dropout in India presents a rather major obstacle in India for equal and inclusive education. Though this phenomenon is complicated and multi-casual, dropout rates are especially high at the secondary level. And typically the policy making does not really solve this issue. By means of machine learning, this paper offers a proactive web based response to this problem. It entails the creation of a thorough platform like a frontend website employing Streamlit Python Framework as well as strong backend supported by Random Forest and Gradient Boosting classifiers. This website provides a user-friendly interface where users can access and also analyze pre-trained model's results on the dashboard while exploring detailed analytics by different factors such as school, age, area and caste. This paper demonstrates how prediction analysis can be transformed into an actionable tool that helps people understand the reasons why student dropout. 
    
    High dropout rates in school education are a major concern, especially due to poverty, social, and economic factors. This system helps the Government of Karnataka identify and address dropout trends across various categories:

- School-wise
- Area-wise (Urban/Rural)
- Gender-wise
- Caste-wise
- Age/Standard-wise

    ### 🎯 Objectives
    Our project aims to:

- Collect and pre-process dropout data from authentic sources.
- Analyse dropout patterns using machine learning models.
- Provide real-time analysis through a web-based dashboard.
- Enable manual data entry via web forms for live monitoring.
- Support policymakers and school authorities in identifying high-risk categories.
- Facilitate focused interventions to reduce dropout rates.
    
    ### 🤖 Technology Stack
    - **Machine Learning**: Random Forest Classifier
    - **Data Processing**: Pandas, NumPy, Scikit-learn
    - **Visualization**: Plotly, Streamlit
    - **Deployment**: Python, Streamlit Cloud
    
    ### 📊 Model Performance
    """)
    
    metrics = load_model_metrics()
    
    if metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{metrics['test_accuracy']*100:.2f}%")
        with col2:
            st.metric("AUC Score", f"{metrics['test_auc']:.3f}")
        with col3:
            st.metric("Model", "Random Forest")
    
    st.markdown("""
    ### 📈 Key Features
    - **Real-time Predictions**: Instant dropout risk assessment
    - **Visual Analytics**: Interactive charts and dashboards
    - **Risk Categorization**: High, Medium, Low risk levels
    - **Actionable Insights**: Personalized recommendations
    - **Feature Importance**: Understand key risk factors
    
    ### 👥 Team
    CBD_7 - Capstone Project Team: 
    Tanushree R, 
    Kavya J, 
    Kavya S.
    
    ### 📧 Contact
    For more information, visit the Contact page.
    """)
    
    # Feature Importance Chart
    st.markdown("---")
    st.subheader("🔍 Model Feature Importance")
    
    feature_imp = load_feature_importance()
    if not feature_imp.empty:
        fig = px.bar(
            feature_imp.sort_values('importance', ascending=True).tail(10),
            x='importance',
            y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale='Viridis',
            title="Top 10 Most Important Features"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# Contact Page
def contact_page():
    st.title("📧 Contact Us")
    
    st.markdown("""
    ### Get in Touch
    
    Have questions or feedback? We'd love to hear from you!
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("contact_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            subject = st.selectbox("Subject", [
                "General Inquiry",
                "Technical Support",
                "Feature Request",
                "Bug Report",
                "Collaboration"
            ])
            message = st.text_area("Message", height=150)
            
            submitted = st.form_submit_button("Send Message", type="primary")
            
            if submitted:
                st.success("✅ Thank you! Your message has been sent. We'll get back to you soon.")
    
    with col2:
        st.markdown("""
        ### 📍 Contact Information
        
        **Roll No:**  
        20221CBD0029

        20221CBD0023

        20221CBD0021

        
        **Mail:**  
        TANUSHREE.20221CBD0029@PRESIDENCYUNIVERSITY.IN
        KAVYA.20221CBD0023@PRESIDENCYUNIVERSITY.IN
        KAVYA.20221CBD0021@PRESIDENCYUNIVERSITY.IN
        
        **Address:**  
        PRESIDENCY UNIVERSITY  
        YELAHANKA  
        560064
        
        ---
        
        ### 🌐 Connect With Us
        - [LinkedIn](#)
        - [GitHub](#)
        - [Twitter](#)
        """)

# Main App
def main():
    page = sidebar()
    
    if "Dashboard" in page:
        dashboard_page()
    elif "Predict" in page:
        predict_page()
    elif "Analytics" in page:
        analytics_page()
    elif "About" in page:
        about_page()
    elif "Contact" in page:
        contact_page()

if __name__ == "__main__":
    main()