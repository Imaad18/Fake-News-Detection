import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import joblib
import os
import base64
from io import BytesIO
import time

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
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        return True
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {str(e)}")
        return False

# Initialize lemmatizer and stopwords
if download_nltk_resources():
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
else:
    lemmatizer = None
    stop_words = set()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F7FF;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    .metric-card {
        background-color: #F0F7FF;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #4B5563;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .fake-tag {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .real-tag {
        background-color: #D1FAE5;
        color: #047857;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Preprocess text: remove special characters, stopwords, and lemmatize
def preprocess_text(text, remove_stopwords=True):
    # Basic cleaning
    text = re.sub(r'\W', ' ', str(text))  # remove non-word characters
    text = re.sub(r'\s+', ' ', text)      # replace multiple spaces with one
    text = text.lower().strip()           # convert to lowercase and trim
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords and lemmatize
    if remove_stopwords:
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 1]
    else:
        words = [lemmatizer.lemmatize(word) for word in words if len(word) > 1]
    
    return ' '.join(words)

# Function to extract text features
def extract_text_features(text):
    features = {}
    
    # Word count
    words = text.split()
    features['word_count'] = len(words)
    
    # Average word length
    if len(words) > 0:
        features['avg_word_length'] = sum(len(word) for word in words) / len(words)
    else:
        features['avg_word_length'] = 0
    
    # Sentence count
    sentences = re.split(r'[.!?]+', text)
    features['sentence_count'] = len([s for s in sentences if len(s.strip()) > 0])
    
    # Average sentence length
    if features['sentence_count'] > 0:
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
    else:
        features['avg_sentence_length'] = 0
    
    return features

# Function to predict on new text
def predict_news(text, model, vectorizer):
    # Preprocess the text
    clean_text = preprocess_text(text)
    
    # Transform the text using the vectorizer
    text_vector = vectorizer.transform([clean_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0][1]  # Probability of being fake
    
    return prediction, probability, clean_text

# Function to get example dataset
@st.cache_data
def get_example_dataset():
    # URL for the dataset
    url = "https://raw.githubusercontent.com/KailasMahavarkar/interview-questions/main/land-data.csv"
    try:
        df = pd.read_csv(url)
        # Create a simple fake news dataset
        data = {
            'title': [
                'Scientists discover new species of dinosaur',
                'Breaking: Aliens contact Earth government',
                'Study shows chocolate is healthier than vegetables',
                'New study links coffee consumption to longevity',
                'Global temperatures hit record high in April',
                'Man claims to have found Bigfoot in national park',
                'Doctors recommend avoiding all sugar consumption',
                'NASA discovers water on Mars surface',
                'Ancient pyramid discovered under Antarctic ice',
                'Local restaurant closed for health violations'
            ],
            'text': [
                'Paleontologists from Oxford University have discovered a new species of dinosaur in Argentina. The fossils indicate it was a herbivore that lived approximately 80 million years ago.',
                'Government officials have confirmed contact with extraterrestrial beings from Alpha Centauri. The aliens are reportedly peaceful and wish to share advanced technology.',
                'A controversial new study claims that chocolate contains more antioxidants and beneficial compounds than most vegetables. Health experts are skeptical of these findings.',
                'Research published in the Journal of Internal Medicine suggests that drinking 3-4 cups of coffee daily is associated with a 15% reduction in all-cause mortality.',
                'Climate scientists report that global average temperatures in April were the highest ever recorded for that month, continuing a concerning warming trend.',
                'A hiker in Yellowstone National Park claims to have captured clear footage of Bigfoot. Wildlife experts dispute the claim, suggesting it's likely a bear.',
                'A group of doctors is recommending that people eliminate all forms of sugar from their diet, claiming it's the only way to prevent diabetes and obesity.',
                'NASA's rover has confirmed the presence of liquid water on the Martian surface, a significant discovery that increases the possibility of finding evidence of life.',
                'Researchers using ground-penetrating radar have allegedly discovered a massive pyramid structure beneath Antarctic ice, suggesting advanced ancient civilizations.',
                'Health inspectors found numerous violations including rodent infestations and improper food storage at popular local restaurant "Tasty Bites." The establishment will remain closed until issues are resolved.'
            ],
            'label': ['true', 'fake', 'fake', 'true', 'true', 'fake', 'fake', 'true', 'fake', 'true'],
            'date': pd.date_range(start='1/1/2023', periods=10).tolist(),
            'category': ['Science', 'Politics', 'Health', 'Health', 'Environment', 'Entertainment', 'Health', 'Science', 'History', 'Local']
        }
        example_df = pd.DataFrame(data)
        return example_df
    except Exception as e:
        st.error(f"Error loading example dataset: {str(e)}")
        return None

# Function to load pre-trained model
@st.cache_resource
def load_pretrained_model():
    try:
        # Initialize and train a simple model with example data
        df = get_example_dataset()
        
        # Prepare data
        X_text = [preprocess_text(text) for text in df['text']]
        y = np.array([1 if label.lower() == 'fake' else 0 for label in df['label']])
        
        # Create and fit vectorizer
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(X_text)
        
        # Train model
        model = LogisticRegression(C=1.0, solver='liblinear', max_iter=1000)
        model.fit(X, y)
        
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading pre-trained model: {str(e)}")
        return None, None

# Function to generate a downloadable DataFrame
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="btn">{text}</a>'
    return href

# Function to create a downloadable plot
def get_figure_download_link(fig, filename="plot.png", text="Download Plot"):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}" class="btn">{text}</a>'
    return href

# Function to analyze text sentiment and credibility
def analyze_text_credibility(text):
    # Number of sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    
    # Word count
    words = text.split()
    
    # Average sentence length
    avg_sentence_len = len(words) / len(sentences) if len(sentences) > 0 else 0
    
    # Capitalized words (excluding first words of sentences)
    caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    caps_ratio = caps_words / len(words) if len(words) > 0 else 0
    
    # Exclamation marks
    exclamation_count = text.count('!')
    
    # Question marks
    question_count = text.count('?')
    
    # URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    url_count = len(url_pattern.findall(text))
    
    # Create credibility metrics
    metrics = {
        'Word Count': len(words),
        'Sentence Count': len(sentences),
        'Avg Words per Sentence': round(avg_sentence_len, 1),
        'ALL CAPS Words': caps_words,
        'ALL CAPS Ratio': round(caps_ratio * 100, 1),
        'Exclamation Marks': exclamation_count,
        'Question Marks': question_count,
        'URLs': url_count
    }
    
    # Create risk factors
    risk_factors = []
    if avg_sentence_len > 25:
        risk_factors.append("Very long sentences (may indicate complex or convoluted arguments)")
    if avg_sentence_len < 10:
        risk_factors.append("Very short sentences (may indicate simplistic arguments)")
    if caps_ratio > 0.1:
        risk_factors.append("Excessive use of ALL CAPS (may be attempting to provoke emotion)")
    if exclamation_count > 3:
        risk_factors.append("Multiple exclamation marks (may indicate emotional manipulation)")
    if url_count == 0:
        risk_factors.append("No source URLs (may lack references)")
    
    return metrics, risk_factors

# Main app title
st.markdown('<h1 class="main-header">📰 Fake News Detection System</h1>', unsafe_allow_html=True)

# Load pre-trained model
model, vectorizer = load_pretrained_model()

# Create tabs
tabs = st.tabs([
    "🔍 News Analyzer", 
    "📊 Dataset Explorer", 
    "📝 Batch Processing",
    "🧠 Educational Resources"
])

with tabs[0]:  # News Analyzer
    st.markdown('<h2 class="subheader">News Text Analysis</h2>', unsafe_allow_html=True)
    
    # User input
    news_text = st.text_area(
        "Paste news article text here for analysis:",
        height=200,
        placeholder="Enter or paste news article text here..."
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        analyze_btn = st.button("Analyze Text", type="primary", use_container_width=True)
    
    with col2:
        # Example selector
        examples = ["Select an example...", "Climate Change Report", "Celebrity Scandal", "Medical Breakthrough", "Political Allegations"]
        selected_example = st.selectbox("Or try an example:", examples)
        
        example_texts = {
            "Climate Change Report": "New scientific research confirms that global temperatures have risen by 1.1°C since pre-industrial times. The study, published in Nature Climate Science, analyzed data from over 3,000 weather stations worldwide and found that the rate of warming has accelerated in the past decade. Lead researcher Dr. Emily Chen stated that immediate action is needed to prevent catastrophic impacts.",
            "Celebrity Scandal": "SHOCKING NEWS!!! Famous actor caught in MASSIVE SCANDAL!! Anonymous sources claim the A-list celebrity has been LYING to fans for YEARS! The star's representatives REFUSE to comment on these allegations!!! Is this the END of their career?? Share this BREAKING story now!!!",
            "Medical Breakthrough": "Scientists at Stanford University have developed a new treatment for Alzheimer's disease that shows promise in early clinical trials. The therapy, which targets protein aggregates in the brain, improved cognitive function in 68% of participants. However, researchers caution that larger studies are needed before the treatment can be approved for general use.",
            "Political Allegations": "CORRUPT POLITICIAN EXPOSED! Secret documents reveal the senator has been accepting illegal donations for years! The mainstream media is COVERING IT UP! They don't want you to know the TRUTH! Wake up, people!!! This is just the tip of the iceberg!!!"
        }
        
        if selected_example != "Select an example...":
            news_text = example_texts[selected_example]
            analyze_btn = True
    
    if analyze_btn and news_text:
        with st.spinner("Analyzing text..."):
            # Extract text features
            features = extract_text_features(news_text)
            
            # Analyze credibility
            metrics, risk_factors = analyze_text_credibility(news_text)
            
            # Make prediction
            prediction, probability, clean_text = predict_news(news_text, model, vectorizer)
            
            # Display results
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown('<h3 class="subheader">Prediction Result</h3>', unsafe_allow_html=True)
                
                if prediction == 1:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <div style="font-size: 3rem; margin-right: 1rem;">🚨</div>
                        <div>
                            <div style="font-size: 1.8rem; font-weight: bold; color: #B91C1C;">Likely Fake News</div>
                            <div style="font-size: 1rem; color: #4B5563;">Confidence: {probability:.1%}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <div style="font-size: 3rem; margin-right: 1rem;">✅</div>
                        <div>
                            <div style="font-size: 1.8rem; font-weight: bold; color: #047857;">Likely Credible News</div>
                            <div style="font-size: 1rem; color: #4B5563;">Confidence: {1-probability:.1%}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.progress(probability if prediction == 1 else 1-probability)
                
                st.markdown('<h3 class="subheader">Risk Factors</h3>', unsafe_allow_html=True)
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(factor)
                else:
                    st.success("No significant risk factors detected in this text.")
            
            with col2:
                st.markdown('<h3 class="subheader">Text Metrics</h3>', unsafe_allow_html=True)
                
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                st.markdown('<h3 class="subheader">Word Cloud</h3>', unsafe_allow_html=True)
                
                # Create word cloud
                try:
                    from wordcloud import WordCloud
                    
                    wordcloud = WordCloud(width=400, height=200, background_color='white', 
                                         colormap='viridis', max_words=50).generate(clean_text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                except ImportError:
                    st.info("WordCloud package not available. Install it for word cloud visualization.")
            
            # Show processing details
            with st.expander("See text processing details"):
                st.markdown("#### Original Text")
                st.write(news_text)
                
                st.markdown("#### Preprocessed Text")
                st.write(clean_text)
                
                st.markdown("#### Common Signs of Fake News")
                st.markdown("""
                - Excessive use of ALL CAPS and exclamation points (!!!)
                - Emotional and sensationalist language
                - Absence of sources or references
                - Extreme claims without specific evidence
                - Unidentified or anonymous sources
                - Spelling and grammatical errors
                - Appeal to confirm existing biases
                """)

with tabs[1]:  # Dataset Explorer
    st.markdown('<h2 class="subheader">Dataset Explorer</h2>', unsafe_allow_html=True)
    
    # File uploader or use example dataset
    st.markdown("Upload your own fake news dataset or use our example dataset:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV file (must contain 'text' and 'label' columns)", type=["csv"])
    
    with col2:
        use_example = st.checkbox("Use example dataset", value=not bool(uploaded_file))
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'label' not in df.columns or 'text' not in df.columns:
                st.error("Uploaded file must contain 'text' and 'label' columns")
                df = None
            else:
                st.success(f"Dataset loaded successfully: {df.shape[0]} rows and {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            df = None
    elif use_example:
        df = get_example_dataset()
        if df is not None:
            st.success(f"Example dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    else:
        df = None
        st.info("Please upload a dataset or select 'Use example dataset'")
    
    if df is not None:
        # Add additional features for analysis
        if 'title' in df.columns:
            df['title_length'] = df['title'].apply(lambda x: len(str(x)))
        df['text_length'] = df['text'].apply(lambda x: len(str(x)))
        
        # Create tabs for different analyses
        data_tabs = st.tabs(["Dataset Overview", "Visualizations", "Text Analysis"])
        
        with data_tabs[0]:  # Dataset Overview
            st.markdown('<h3 class="subheader">Dataset Preview</h3>', unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">'
                           f'<div class="metric-value">{df.shape[0]}</div>'
                           f'<div class="metric-label">Total Records</div>'
                           '</div>', unsafe_allow_html=True)
            
            with col2:
                label_counts = df['label'].value_counts()
                fake_count = label_counts.get('fake', 0) if isinstance(label_counts.index[0], str) else label_counts.get(1, 0)
                st.markdown('<div class="metric-card">'
                           f'<div class="metric-value">{fake_count}</div>'
                           f'<div class="metric-label">Fake News Articles</div>'
                           '</div>', unsafe_allow_html=True)
            
            with col3:
                real_count = label_counts.get('true', 0) if isinstance(label_counts.index[0], str) else label_counts.get(0, 0)
                st.markdown('<div class="metric-card">'
                           f'<div class="metric-value">{real_count}</div>'
                           f'<div class="metric-label">Real News Articles</div>'
                           '</div>', unsafe_allow_html=True)
            
            st.markdown('<h3 class="subheader">Column Information</h3>', unsafe_allow_html=True)
            
            # Column info
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
            
            # Download dataset
            st.markdown(get_download_link(df, "fake_news_dataset.csv", "Download Dataset"), unsafe_allow_html=True)
        
        with data_tabs[1]:  # Visualizations
            st.markdown('<h3 class="subheader">Data Visualizations</h3>', unsafe_allow_html=True)
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Category Distribution
                if 'category' in df.columns:
                    st.markdown("#### News Category Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    category_counts = df['category'].value_counts()
                    category_counts.plot(kind='bar', ax=ax, color='skyblue')
                    plt.title('News by Category')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Download option
                    st.markdown(get_figure_download_link(fig, "category_distribution.png", "Download Plot"), 
                               unsafe_allow_html=True)
                
                # Text Length Distribution
                st.markdown("#### Text Length Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Group by label and plot histograms
                for label in df['label'].unique():
                    subset = df[df['label'] == label]
                    sns.histplot(data=subset, x='text_length', kde=True, 
                                 alpha=0.5, label=label, ax=ax)
                
                plt.title('Text Length Distribution by Label')
                plt.xlabel('Text Length (characters)')
                plt.legend()
                plt.tight_layout()
                st.pyplot(fig)
                
                # Download option
                st.markdown(get_figure_download_link(fig, "text_length_distribution.png", "Download Plot"), 
                           unsafe_allow_html=True)
            
            with viz_col2:
                # Label Distribution
                st.markdown("#### News Label Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                label_counts = df['label'].value_counts()
                colors = ['#1E3A8A', '#B91C1C'] if len(label_counts) == 2 else None
                label_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=colors)
                plt.title('Distribution of Fake vs Real News')
                plt.ylabel('')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Download option
                st.markdown(get_figure_download_link(fig, "label_distribution.png", "Download Plot"), 
                           unsafe_allow_html=True)
                
                if 'date' in df.columns:
                    # Try to convert to datetime if not already
                    if df['date'].dtype != 'datetime64[ns]':
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    
                    if not df['date'].isna().all():
                        st.markdown("#### Publication Trends")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Group by month and label
                        df['month'] = df['date'].dt.to_period('M')
                        monthly_counts = df.groupby(['month', 'label']).size().unstack()
                        
                        # Plot
                        monthly_counts.plot(kind='line', marker='o', ax=ax)
                        plt.title('News Publication Trends by Month')
                        plt.xlabel('Month')
                        plt.ylabel('Count')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Download option
                        st.markdown(get_figure_download_link(fig, "publication_trends.png", "Download Plot"), 
                                   unsafe_allow_html=True)
        
        with data_tabs[2]:  # Text Analysis
            st.markdown('<h3 class="subheader">Text Content Analysis</h3>', unsafe_allow_html=True)
            
            # Sample size
            sample_size = min(1000, len(df))
            text_sample = df.sample(sample_size)
            
            # Process text sample
            st.markdown(f"Analyzing a sample of {sample_size} articles...")
            
            with st.spinner("Processing text..."):
                # Preprocess text
                text_sample['clean_text'] = text_sample['text'].apply(lambda x: preprocess_text(str(x)))
                
                # Get word counts
                all_words = ' '.join(text_sample['clean_text']).split()
                word_counts = pd.Series(all_words).value_counts()
                
                # Split by label
                fake_words = ' '.join(text_sample[text_sample['label'] == 'fake']['clean_text']).split()
                real_words = ' '.join(text_sample[text_sample['label'] == 'true']['clean_text']).split()
                
                fake_word_counts = pd.Series(fake_words).value_counts()
                real_word_counts = pd.Series(real_words).value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Most Common Words")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                top_words = word_counts.head(20)
                top_words.plot(kind='barh', ax=ax, color='skyblue')
                plt.title('Top 20 Words in News Articles')
                plt.xlabel('Count')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Download option
                st.markdown(get_figure_download_link(fig, "top_words.png", "Download Plot"), 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Word Clouds by Label")
                
                # Create tabs for fake and real news word clouds
                wc_tabs = st.tabs(["Fake News", "Real News"])
                
                try:
                    from wordcloud import WordCloud
                    
                    with wc_tabs[0]:
                        if len(fake_words) > 0:
                            fake_text = ' '.join(fake_words)
                            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                                colormap='Reds', max_words=100).generate(fake_text)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            plt.title('Fake News Word Cloud')
                            st.pyplot(fig)
                        else:
                            st.info("No fake news articles in the sample.")
                    
                    with wc_tabs[1]:
                        if len(real_words) > 0:
                            real_text = ' '.join(real_words)
                            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                                colormap='Blues', max_words=100).generate(real_text)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            plt.title('Real News Word Cloud')
                            st.pyplot(fig)
                        else:
                            st.info("No real news articles in the sample.")
                except ImportError:
                    st.error("WordCloud package not installed. Install it for word cloud visualization.")
            
            # N-gram analysis
            st.markdown("#### N-gram Analysis")
            ngram_type = st.radio("Select n-gram type:", ["Unigrams (1-gram)", "Bigrams (2-gram)", "Trigrams (3-gram)"], horizontal=True)
            n = 1 if "Unigrams" in ngram_type else (2 if "Bigrams" in ngram_type else 3)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"##### Top {n}-grams in Fake News")
                if len(fake_words) >= n:
                    fake_ngrams = pd.Series(nltk.ngrams(fake_words, n)).value_counts().head(10)
                    st.dataframe(fake_ngrams, use_container_width=True)
                else:
                    st.info(f"Not enough fake news articles for {n}-gram analysis")
            
            with col2:
                st.markdown(f"##### Top {n}-grams in Real News")
                if len(real_words) >= n:
                    real_ngrams = pd.Series(nltk.ngrams(real_words, n)).value_counts().head(10)
                    st.dataframe(real_ngrams, use_container_width=True)
                else:
                    st.info(f"Not enough real news articles for {n}-gram analysis")

with tabs[2]:  # Batch Processing
    st.markdown('<h2 class="subheader">Batch Processing</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload a CSV file containing multiple news articles to analyze them in bulk. 
    The file must contain a column named 'text' with the news content.
    """)
    
    uploaded_batch = st.file_uploader("Upload CSV file for batch processing", type=["csv"])
    
    if uploaded_batch is not None:
        try:
            batch_df = pd.read_csv(uploaded_batch)
            
            if 'text' not in batch_df.columns:
                st.error("Uploaded file must contain a 'text' column")
            else:
                st.success(f"File loaded successfully: {batch_df.shape[0]} articles")
                
                # Process in batches
                if st.button("Analyze Batch", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    total = len(batch_df)
                    
                    for i, row in batch_df.iterrows():
                        text = str(row['text'])
                        
                        # Make prediction
                        prediction, probability, _ = predict_news(text, model, vectorizer)
                        
                        results.append({
                            'text': text[:200] + "..." if len(text) > 200 else text,
                            'prediction': 'Fake' if prediction == 1 else 'Real',
                            'probability': probability if prediction == 1 else 1-probability,
                            'confidence': 'High' if (probability if prediction == 1 else 1-probability) > 0.75 else 'Medium' if (probability if prediction == 1 else 1-probability) > 0.5 else 'Low'
                        })
                        
                        # Update progress
                        progress = (i + 1) / total
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {i+1} of {total} articles...")
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(results)
                    
                    # Show results
                    st.markdown('<h3 class="subheader">Batch Results</h3>', unsafe_allow_html=True)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown('<h3 class="subheader">Summary Statistics</h3>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        fake_count = results_df[results_df['prediction'] == 'Fake'].shape[0]
                        st.markdown('<div class="metric-card">'
                                   f'<div class="metric-value">{fake_count}</div>'
                                   f'<div class="metric-label">Fake News</div>'
                                   '</div>', unsafe_allow_html=True)
                    
                    with col2:
                        real_count = results_df[results_df['prediction'] == 'Real'].shape[0]
                        st.markdown('<div class="metric-card">'
                                   f'<div class="metric-value">{real_count}</div>'
                                   f'<div class="metric-label">Real News</div>'
                                   '</div>', unsafe_allow_html=True)
                    
                    with col3:
                        avg_prob = results_df['probability'].mean()
                        st.markdown('<div class="metric-card">'
                                   f'<div class="metric-value">{avg_prob:.1%}</div>'
                                   f'<div class="metric-label">Avg Confidence</div>'
                                   '</div>', unsafe_allow_html=True)
                    
                    # Download results
                    st.markdown(get_download_link(results_df, "batch_results.csv", "Download Results"), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing batch file: {str(e)}")

with tabs[3]:  # Educational Resources
    st.markdown('<h2 class="subheader">Educational Resources</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### How to Identify Fake News
    
    Fake news can be difficult to spot, but there are several signs to look for:
    """)
    
    with st.expander("1. Check the Source"):
        st.markdown("""
        - Investigate the website, its mission, and contact info
        - Look for unusual URLs or site names
        - Check the "About Us" section for transparency
        - See if other reputable news sources are reporting the same story
        """)
    
    with st.expander("2. Examine the Evidence"):
        st.markdown("""
        - Credible stories will include facts, data, and expert quotes
        - Look for original sources and verify they support the claims
        - Be skeptical of anonymous sources or lack of sources
        - Check if images are authentic (reverse image search can help)
        """)
    
    with st.expander("3. Watch for Emotional Language"):
        st.markdown("""
        - Sensationalist or emotionally charged language is a red flag
        - Excessive use of ALL CAPS or exclamation points!!!
        - Headlines that seem too good (or bad) to be true
        - Content designed to provoke anger or fear
        """)
    
    with st.expander("4. Check the Date and Context"):
        st.markdown("""
        - Old stories may be shared as if they're new
        - Satirical content may be mistaken for real news
        - Verify dates on photos, videos, and social media posts
        - Consider if the story makes sense in current context
        """)
    
    with st.expander("5. Review Your Own Biases"):
        st.markdown("""
        - We're more likely to believe stories that confirm our existing views
        - Be extra skeptical of content that aligns perfectly with your beliefs
        - Fact-check information before sharing, even if it supports your opinion
        """)
    
    st.markdown("""
    ### Additional Resources
    
    - [International Fact-Checking Network](https://www.poynter.org/ifcn/)
    - [Snopes](https://www.snopes.com/) - The oldest and largest fact-checking site
    - [FactCheck.org](https://www.factcheck.org/) - A nonpartisan fact-checking website
    - [Google Fact Check Explorer](https://toolbox.google.com/factcheck/explorer)
    - [Media Bias/Fact Check](https://mediabiasfactcheck.com/) - Assesses bias and reliability
    """)
    
    st.markdown("""
    ### About This Tool
    
    This Fake News Detection System uses machine learning to analyze news content and assess its credibility. 
    The model examines linguistic patterns, writing style, and content features that are often associated 
    with misinformation.
    
    **Note:** This tool provides an automated analysis but should not be the sole factor in determining 
    credibility. Always verify information through multiple reliable sources.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>Fake News Detection System • Powered by Machine Learning • For educational purposes only</p>
</div>
""", unsafe_allow_html=True)
