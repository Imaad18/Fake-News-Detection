import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('wordnet')
    nltk.download('omw-1.4')

download_nltk_resources()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess text: remove special characters and lemmatize
def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))  # remove non-word characters
    text = re.sub(r'\s+', ' ', text)      # replace multiple spaces with one
    text = text.lower()                   # convert to lowercase
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if len(word) > 1])  # Lemmatization
    return text

# Function to train model and save it
@st.cache_resource
def train_model(df):
    # Preprocess the text column
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    # Encode the target variable
    df['label_encoded'] = df['label'].str.lower().map({'fake': 1, 'true': 0})
    
    # Prepare the text data using TF-IDF (with ngrams)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df['clean_text'])
    
    # Define features and target
    X = X_text  
    y = df['label_encoded']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train logistic regression model
    model = LogisticRegression(C=10, solver='liblinear', max_iter=2000)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate ROC curve data
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'roc_data': (fpr, tpr, roc_auc),
        'X_test': X_test,
        'y_test': y_test
    }

# Function to predict on new text
def predict_news(text, model, vectorizer):
    # Preprocess the text
    clean_text = preprocess_text(text)
    
    # Transform the text using the trained vectorizer
    text_vector = vectorizer.transform([clean_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0][prediction]
    
    return prediction, probability

# Title and description
st.title("📰 Fake News Detection App")
st.markdown("""
This application allows you to:
- Upload and analyze a fake news dataset
- Visualize data characteristics
- Train and evaluate a machine learning model
- Predict whether a news article is fake or real
""")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a page", ["Data Analysis", "Model Training & Evaluation", "Prediction"])

# Upload dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload fake news CSV dataset", type=["csv"])

model_results = None
df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Basic data cleaning
        df = df.dropna(subset=['label', 'text'])
        df = df.drop_duplicates(subset=['text'])
        
        # Create additional features if they don't exist
        if 'title' in df.columns:
            df['title_length'] = df['title'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        df['text_length'] = df['text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
        
        # Convert date if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        # Train the model if data is available
        if st.sidebar.button("Train Model"):
            with st.spinner("Training model... This might take a few minutes."):
                model_results = train_model(df)
            st.sidebar.success("Model trained successfully!")
    
    except Exception as e:
        st.error(f"Error loading or processing the dataset: {str(e)}")

# Content based on selected page
if page == "Data Analysis" and df is not None:
    st.header("Data Analysis")
    
    # Display the first few rows
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Dataset info
    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Rows", df.shape[0])
    col2.metric("Number of Columns", df.shape[1])
    col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column info
    st.subheader("Column Details")
    st.dataframe(pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum()
    }))
    
    # Visualizations
    st.subheader("Visualizations")
    
    # Category distribution if exists
    if 'category' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=df, x='category', order=df['category'].value_counts().index, ax=ax)
        plt.title('News Category Counts')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Label distribution
    if 'label' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x='label', ax=ax)
        plt.title('Label Distribution (Fake vs True)')
        st.pyplot(fig)
    
    # Text length distribution
    st.subheader("Text Length Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'title' in df.columns:
        sns.histplot(df['title_length'], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title('Distribution of Title Length')
    
    sns.histplot(df['text_length'], kde=True, ax=axes[1], color='salmon')
    axes[1].set_title('Distribution of Text Length')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Box plots
    st.subheader("Length Box Plots")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'title' in df.columns:
        sns.boxplot(y=df['title_length'], ax=axes[0], color='lightgreen')
        axes[0].set_title('Title Length')
    
    sns.boxplot(y=df['text_length'], ax=axes[1], color='lightpink')
    axes[1].set_title('Text Length')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Correlation heatmap if we have multiple numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        plt.title('Correlation Heatmap')
        st.pyplot(fig)

elif page == "Model Training & Evaluation":
    st.header("Model Training & Evaluation")
    
    if df is None:
        st.warning("Please upload a dataset first.")
    elif model_results is None:
        st.info("Click 'Train Model' in the sidebar to train the model.")
    else:
        # Display model performance
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{model_results['accuracy']:.4f}")
        
        report = model_results['classification_report']
        col2.metric("F1-Score (Fake News)", f"{report['1']['f1-score']:.4f}")
        
        # Classification report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(model_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
        
        # ROC Curve
        st.subheader("ROC Curve")
        fpr, tpr, roc_auc = model_results['roc_data']
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        st.pyplot(fig)

elif page == "Prediction":
    st.header("Predict Fake News")
    
    if model_results is None:
        st.warning("Please upload a dataset and train the model first.")
    else:
        # Input text area
        news_text = st.text_area("Enter news article text", height=250)
        
        if st.button("Predict"):
            if news_text:
                # Make prediction
                prediction, probability = predict_news(news_text, model_results['model'], model_results['vectorizer'])
                
                # Display result
                if prediction == 1:
                    st.error(f"Prediction: FAKE NEWS (Confidence: {probability:.2%})")
                else:
                    st.success(f"Prediction: REAL NEWS (Confidence: {probability:.2%})")
                
                # Display preprocessed text
                with st.expander("View Preprocessed Text"):
                    st.text(preprocess_text(news_text))
            else:
                st.warning("Please enter some text to analyze.")
else:
    if df is None:
        st.info("Please upload a dataset to get started.")

# Footer
st.markdown("---")
st.markdown("Fake News Detection App - Built with Streamlit")
