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
import joblib
import base64
from io import BytesIO
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

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
        st.warning(f"Failed to download NLTK resources: {str(e)}. Some features may be limited.")
        return False

# Initialize lemmatizer and stopwords
nltk_download_success = download_nltk_resources()
if nltk_download_success:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
else:
    lemmatizer = None
    stop_words = set()

# Inject modern CSS styling
def inject_css():
    st.markdown("""
    <style>
        /* Base styles */
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: #1a1a1a;
        }
        
        /* Header styles */
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subheader {
            font-size: 1.5rem;
            color: #1E3A8A;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        
        /* Tab styles */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            padding: 0 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #F8FAFC;
            border-radius: 12px;
            gap: 1px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.2s ease;
            border: 1px solid #E2E8F0;
            margin: 4px 0;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #EFF6FF;
            transform: translateY(-2px);
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1E3A8A;
            color: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Card styles */
        .metric-card {
            background-color: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid #E2E8F0;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #64748B;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #1E3A8A;
            line-height: 1.2;
        }
        
        /* Tag styles */
        .fake-tag {
            background-color: #FEE2E2;
            color: #B91C1C;
            padding: 0.375rem 0.75rem;
            border-radius: 8px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            font-size: 0.875rem;
        }
        
        .real-tag {
            background-color: #D1FAE5;
            color: #047857;
            padding: 0.375rem 0.75rem;
            border-radius: 8px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            font-size: 0.875rem;
        }
        
        /* Button styles */
        .stButton>button {
            border-radius: 12px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s ease;
            border: none;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #1E3A8A;
            border-radius: 12px;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Text input */
        .stTextArea textarea {
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid #E2E8F0;
        }
        
        /* Selectbox */
        .stSelectbox select {
            border-radius: 12px;
            padding: 0.5rem 1rem;
        }
        
        /* Radio buttons */
        .stRadio [role="radiogroup"] {
            gap: 1rem;
        }
        
        .stRadio [role="radio"] {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
        }
        
        .stRadio [role="radio"][aria-checked="true"] {
            background-color: #EFF6FF;
            border-color: #1E3A8A;
            color: #1E3A8A;
            font-weight: 500;
        }
        
        /* Expander */
        .stExpander {
            border-radius: 12px;
            border: 1px solid #E2E8F0;
        }
        
        /* Custom button for downloads */
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.625rem 1.25rem;
            background-color: #1E3A8A;
            color: white;
            text-decoration: none;
            border-radius: 12px;
            margin-top: 0.5rem;
            font-weight: 600;
            transition: all 0.2s ease;
            border: none;
            cursor: pointer;
            font-size: 0.875rem;
        }
        
        .btn:hover {
            background-color: #3B82F6;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Word cloud container */
        .wordcloud-container {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid #E2E8F0;
        }
        
        /* Better visibility for text */
        .stMarkdown p, .stMarkdown li, .stMarkdown ol {
            color: #1a1a1a;
            font-size: 1rem;
            line-height: 1.7;
        }
        
        /* Code blocks */
        .stCodeBlock {
            border-radius: 12px;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #64748B;
            font-size: 0.875rem;
            padding: 1.5rem 0;
            margin-top: 2rem;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .main-header {
                font-size: 2rem;
            }
            
            .subheader {
                font-size: 1.25rem;
            }
            
            .metric-value {
                font-size: 1.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# Call the CSS injection function
inject_css()

# Preprocess text: remove special characters, stopwords, and lemmatize
def preprocess_text(text, remove_stopwords=True):
    if not text or not isinstance(text, str):
        return ""
    # Basic cleaning
    text = re.sub(r'\W', ' ', text)  # remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # replace multiple spaces with one
    text = text.lower().strip()       # convert to lowercase and trim
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords and lemmatize
    if nltk_download_success and remove_stopwords:
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 1]
    else:
        words = [lemmatizer.lemmatize(word) for word in words if len(word) > 1] if nltk_download_success else words
    
    return ' '.join(words)

# Function to extract text features
def extract_text_features(text):
    features = {}
    
    # Word count
    words = text.split()
    features['word_count'] = len(words)
    
    # Average word length
    features['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
    
    # Sentence count
    sentences = re.split(r'[.!?]+', text)
    features['sentence_count'] = len([s for s in sentences if s.strip()])
    
    # Average sentence length
    features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
    
    return features

# Function to predict on new text
def predict_news(text, model, vectorizer):
    if not text or not isinstance(text, str):
        return None, None, ""
    # Preprocess the text
    clean_text = preprocess_text(text)
    if not clean_text:
        return None, None, ""
    
    # Transform the text using the vectorizer
    text_vector = vectorizer.transform([clean_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0][1]  # Probability of being fake
    
    return prediction, probability, clean_text

# Function to get example dataset
@st.cache_data
def get_example_dataset():
    try:
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
                'Paleontologists from Oxford University have discovered a new species of dinosaur in Argentina...',
                'Government officials have confirmed contact with extraterrestrial beings from Alpha Centauri...',
                'A controversial new study claims that chocolate contains more antioxidants...',
                'Research published in the Journal of Internal Medicine suggests that drinking 3-4 cups of coffee...',
                'Climate scientists report that global average temperatures in April were the highest ever recorded...',
                'A hiker in Yellowstone National Park claims to have captured clear footage of Bigfoot...',
                'A group of doctors is recommending that people eliminate all forms of sugar from their diet...',
                'NASA's rover has confirmed the presence of liquid water on the Martian surface...',
                'Researchers using ground-penetrating radar have allegedly discovered a massive pyramid structure...',
                'Health inspectors found numerous violations including rodent infestations...'
            ],
            'label': ['true', 'fake', 'fake', 'true', 'true', 'fake', 'fake', 'true', 'fake', 'true'],
            'date': pd.date_range(start='2023-01-01', periods=10).tolist(),
            'category': ['Science', 'Politics', 'Health', 'Health', 'Environment', 'Entertainment', 'Health', 'Science', 'History', 'Local']
        }
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error loading example dataset: {str(e)}")
        return pd.DataFrame()

# Function to load pre-trained model
@st.cache_resource
def load_pretrained_model():
    try:
        df = get_example_dataset()
        if df.empty:
            raise ValueError("Failed to load example dataset for model training")
        
        X_text = [preprocess_text(text) for text in df['text']]
        y = np.array([1 if label.lower() == 'fake' else 0 for label in df['label']])
        
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(X_text)
        
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
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="btn">{text}</a>'

# Function to create a downloadable plot
def get_figure_download_link(fig, filename="plot.png", text="Download Plot"):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}" class="btn">{text}</a>'

# Function to analyze text sentiment and credibility
def analyze_text_credibility(text):
    if not text or not isinstance(text, str):
        return {}, []
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    
    words = text.split()
    
    avg_sentence_len = len(words) / len(sentences) if sentences else 0
    
    caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    caps_ratio = caps_words / len(words) if words else 0
    
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    url_count = len(url_pattern.findall(text))
    
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
    
    risk_factors = []
    if avg_sentence_len > 25:
        risk_factors.append("Very long sentences (may indicate complex or convoluted arguments)")
    if avg_sentence_len < 10 and avg_sentence_len > 0:
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

if model is None or vectorizer is None:
    st.error("Failed to load the model. Some features may not work.")
    st.stop()

# Create tabs
tabs = st.tabs([
    "🔍 News Analyzer", 
    "📊 Dataset Explorer", 
    "📝 Batch Processing",
    "🧠 Educational Resources"
])

with tabs[0]:  # News Analyzer
    st.markdown('<h2 class="subheader">News Text Analysis</h2>', unsafe_allow_html=True)
    
    news_text = st.text_area(
        "Paste news article text here for analysis:",
        height=200,
        placeholder="Enter or paste news article text here..."
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        analyze_btn = st.button("Analyze Text", type="primary", use_container_width=True)
    
    with col2:
        examples = ["Select an example...", "Climate Change Report", "Celebrity Scandal", "Medical Breakthrough", "Political Allegations"]
        selected_example = st.selectbox("Or try an example:", examples)
        
        example_texts = {
            "Climate Change Report": "New scientific research confirms that global temperatures have risen by 1.1°C since pre-industrial times...",
            "Celebrity Scandal": "SHOCKING NEWS!!! Famous actor caught in MASSIVE SCANDAL!! Anonymous sources claim...",
            "Medical Breakthrough": "Scientists at Stanford University have developed a new treatment for Alzheimer's disease...",
            "Political Allegations": "CORRUPT POLITICIAN EXPOSED! Secret documents reveal the senator has been accepting illegal donations..."
        }
        
        if selected_example != "Select an example...":
            news_text = example_texts[selected_example]
            analyze_btn = True
    
    if analyze_btn and news_text:
        if not news_text.strip():
            st.error("Please enter valid text for analysis.")
        else:
            with st.spinner("Analyzing text..."):
                features = extract_text_features(news_text)
                metrics, risk_factors = analyze_text_credibility(news_text)
                prediction, probability, clean_text = predict_news(news_text, model, vectorizer)
                
                if prediction is None:
                    st.error("Failed to process the text. Please try again.")
                    st.stop()
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown('<h3 class="subheader">Prediction Result</h3>', unsafe_allow_html=True)
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
                            <div style="font-size: 3rem; margin-right: 1.5rem;">🚨</div>
                            <div>
                                <div style="font-size: 1.8rem; font-weight: bold; color: #B91C1C;">Likely Fake News</div>
                                <div style="font-size: 1rem; color: #4B5563;">Confidence: {probability:.1%}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
                            <div style="font-size: 3rem; margin-right: 1.5rem;">✅</div>
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
                    
                    if WORDCLOUD_AVAILABLE and clean_text:
                        wordcloud = WordCloud(width=400, height=200, background_color='white', 
                                             colormap='viridis', max_words=50).generate(clean_text)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        st.info("Word cloud visualization unavailable. Install 'wordcloud' package or ensure valid text input.")
                
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
    
    st.markdown("Upload your own fake news dataset or use our example dataset:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV file (must contain 'text' and 'label' columns)", type=["csv"])
    
    with col2:
        use_example = st.checkbox("Use example dataset", value=not bool(uploaded_file))
    
    df = None
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
        if not df.empty:
            st.success(f"Example dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns")
        else:
            df = None
            st.error("Failed to load example dataset.")
    
    if df is not None:
        if 'title' in df.columns:
            df['title_length'] = df['title'].apply(lambda x: len(str(x)))
        df['text_length'] = df['text'].apply(lambda x: len(str(x)))
        
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
            
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
            
            st.markdown(get_download_link(df, "fake_news_dataset.csv", "Download Dataset"), unsafe_allow_html=True)
        
        with data_tabs[1]:  # Visualizations
            st.markdown('<h3 class="subheader">Data Visualizations</h3>', unsafe_allow_html=True)
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                if 'category' in df.columns:
                    st.markdown("#### News Category Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    category_counts = df['category'].value_counts()
                    category_counts.plot(kind='bar', ax=ax, color='skyblue')
                    plt.title('News by Category')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown(get_figure_download_link(fig, "category_distribution.png", "Download Plot"), 
                               unsafe_allow_html=True)
                
                st.markdown("#### Text Length Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for label in df['label'].unique():
                    subset = df[df['label'] == label]
                    sns.histplot(data=subset, x='text_length', kde=True, 
                                 alpha=0.5, label=label, ax=ax)
                
                plt.title('Text Length Distribution by Label')
                plt.xlabel('Text Length (characters)')
                plt.legend()
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown(get_figure_download_link(fig, "text_length_distribution.png", "Download Plot"), 
                           unsafe_allow_html=True)
            
            with viz_col2:
                st.markdown("#### News Label Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                label_counts = df['label'].value_counts()
                colors = ['#1E3A8A', '#B91C1C'] if len(label_counts) == 2 else None
                label_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=colors)
                plt.title('Distribution of Fake vs Real News')
                plt.ylabel('')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown(get_figure_download_link(fig, "label_distribution.png", "Download Plot"), 
                           unsafe_allow_html=True)
                
                if 'date' in df.columns:
                    if df['date'].dtype != 'datetime64[ns]':
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    
                    if not df['date'].isna().all():
                        st.markdown("#### Publication Trends")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        df['month'] = df['date'].dt.to_period('M')
                        monthly_counts = df.groupby(['month', 'label']).size().unstack()
                        
                        monthly_counts.plot(kind='line', marker='o', ax=ax)
                        plt.title('News Publication Trends by Month')
                        plt.xlabel('Month')
                        plt.ylabel('Count')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.markdown(get_figure_download_link(fig, "publication_trends.png", "Download Plot"), 
                                   unsafe_allow_html=True)
                    else:
                        st.info("No valid date data available for publication trends.")
        
        with data_tabs[2]:  # Text Analysis
            st.markdown('<h3 class="subheader">Text Content Analysis</h3>', unsafe_allow_html=True)
            
            sample_size = min(1000, len(df))
            text_sample = df.sample(sample_size, random_state=42)
            
            st.markdown(f"Analyzing a sample of {sample_size} articles...")
            
            with st.spinner("Processing text..."):
                text_sample['clean_text'] = text_sample['text'].apply(lambda x: preprocess_text(str(x)))
                
                all_words = ' '.join(text_sample['clean_text']).split()
                word_counts = pd.Series(all_words).value_counts()
                
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
                
                st.markdown(get_figure_download_link(fig, "top_words.png", "Download Plot"), 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Word Clouds by Label")
                
                wc_tabs = st.tabs(["Fake News", "Real News"])
                
                if WORDCLOUD_AVAILABLE:
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
                else:
                    st.info("Word cloud visualization unavailable. Install 'wordcloud' package.")
            
            st.markdown("#### N-gram Analysis")
            ngram_type = st.radio("Select n-gram type:", ["Unigrams (1-gram)", "Bigrams (2-gram)", "Trigrams (3-gram)"], horizontal=True)
            n = 1 if "Unigrams" in ngram_type else (2 if "Bigrams" in ngram_type else 3)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"##### Top {n}-grams in Fake News")
                if len(fake_words) >= n:
                    fake_ngrams = pd.Series(nltk.ngrams(fake_words, n)).value_counts().head(10)
                    st.dataframe(pd.DataFrame({'N-gram': [' '.join(ngram) for ngram in fake_ngrams.index], 'Count': fake_ngrams.values}), use_container_width=True)
                else:
                    st.info(f"Not enough fake news articles for {n}-gram analysis")
            
            with col2:
                st.markdown(f"##### Top {n}-grams in Real News")
                if len(real_words) >= n:
                    real_ngrams = pd.Series(nltk.ngrams(real_words, n)).value_counts().head(10)
                    st.dataframe(pd.DataFrame({'N-gram': [' '.join(ngram) for ngram in real_ngrams.index], 'Count': real_ngrams.values}), use_container_width=True)
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
                
                if st.button("Analyze Batch", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    batch_df['clean_text'] = batch_df['text'].apply(lambda x: preprocess_text(str(x)))
                    text_vectors = vectorizer.transform(batch_df['clean_text'])
                    predictions = model.predict(text_vectors)
                    probabilities = model.predict_proba(text_vectors)[:, 1]
                    
                    results = []
                    total = len(batch_df)
                    
                    for i, (text, pred, prob) in enumerate(zip(batch_df['text'], predictions, probabilities)):
                        results.append({
                            'text': text[:200] + "..." if len(text) > 200 else text,
                            'prediction': 'Fake' if pred == 1 else 'Real',
                            'probability': prob if pred == 1 else 1-prob,
                            'confidence': 'High' if (prob if pred == 1 else 1-prob) > 0.75 else 'Medium' if (prob if pred == 1 else 1-prob) > 0.5 else 'Low'
                        })
                        
                        progress = (i + 1) / total
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {i+1} of {total} articles...")
                    
                    results_df = pd.DataFrame(results)
                    
                    st.markdown('<h3 class="subheader">Batch Results</h3>', unsafe_allow_html=True)
                    st.dataframe(results_df, use_container_width=True)
                    
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
<div class="footer">
    <p>Fake News Detection System • Powered by Machine Learning • For educational purposes only</p>
</div>
""", unsafe_allow_html=True)
