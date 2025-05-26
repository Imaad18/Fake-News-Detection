![scrnli_t0fetWomJ0z8Xi](https://github.com/user-attachments/assets/a00701ca-9d1e-4575-bbcb-476f7065c6b8)




# Fake News Detection System

# Overview

* The Fake News Detection System is a web-based application built using Streamlit and powered by machine learning to detect fake news articles.

* It leverages natural language processing (NLP) techniques and a logistic regression model to classify news as either "Fake" or "Real" based on textual content. The application provides an interactive interface for single article analysis, batch processing, dataset exploration, and educational resources to help users understand fake news detection.

* This project is designed for educational purposes, demonstrating how NLP and machine learning can be applied to combat misinformation. It includes visualizations, risk factor analysis, and a user-friendly design to make the tool accessible to both technical and non-technical users.

  
# Features

# 1.News Analyzer:

* Input a news article or select from predefined examples for real-time analysis.

* Predicts whether the article is "Fake" or "Real" with confidence scores.

* Displays text metrics (e.g., word count, sentence length, ALL CAPS ratio).
  
* Identifies risk factors (e.g., sensational language, lack of sources).

* Generates a word cloud visualization (requires wordcloud package).


# 2.Dataset Explorer:

* Upload a CSV dataset or use the built-in example dataset.

* Provides dataset previews, summary statistics, and visualizations:

* News category distribution.

* Text length distribution by label.

* Label distribution (Fake vs. Real).

* Publication trends (if date data is available).

* Supports n-gram analysis and word clouds for fake and real news.


# 3.Batch Processing:

* Upload a CSV file containing multiple news articles for bulk analysis.

* Outputs predictions, confidence scores, and summary statistics.

* Allows downloading results as a CSV file.


# 4.Educational Resources:

* Guides users on identifying fake news (e.g., checking sources, examining evidence).

* Provides links to external fact-checking resources.

* Explains the tool’s methodology and limitations.



# Technologies Used

# Python Libraries:

* streamlit: For the web-based user interface.

* nltk: For text preprocessing (lemmatization, stopwords).

* scikit-learn: For TF-IDF vectorization and logistic regression.

* pandas: For data manipulation.

* matplotlib and seaborn: For visualizations.

* wordcloud (optional): For word cloud visualizations.

* numpy: For numerical operations.

* joblib: For model persistence.

* base64 and io: For file downloads.


# Machine Learning:

* Logistic Regression model trained on TF-IDF features.

* Preprocesses text by removing special characters, lemmatizing, and filtering stopwords.


# Frontend:

* Custom CSS for enhanced UI/UX.
* Streamlit tabs and components for navigation and interactivity.



# Installation

* Prerequisites

* Python 3.8 or higher

* pip package manager

# Steps

Clone the Repository:

```bash
git clone https://github.com/your-username/fake-news-detection-system.git
cd fake-news-detection-system
```

# Create a Virtual Environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

# Install Dependencies:

```bash
pip install -r requirements.txt
```

The requirements.txt should include:

* streamlit==1.38.0

* numpy==1.26.4

* pandas==2.2.2

* matplotlib==3.9.2

* seaborn==0.13.2

* nltk==3.8.1

* scikit-learn==1.5.1

* joblib==1.4.2

* wordcloud==1.9.3


Download NLTK Resources:Run the following Python script to download required NLTK data:
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')


# Run the Application:
streamlit run app.py

The application will open in your default web browser at http://localhost:8501.


# Usage

# News Analyzer:

Paste a news article or select an example from the dropdown.
Click "Analyze Text" to view predictions, metrics, risk factors, and a word cloud.
Expand the "See text processing details" section for preprocessing insights.


# Dataset Explorer:

Upload a CSV file with text and label columns or use the example dataset.
Explore dataset previews, visualizations, and text analysis (e.g., word clouds, n-grams).
Download visualizations or the dataset as needed.


# Batch Processing:

Upload a CSV file with a text column.
Click "Analyze Batch" to process all articles and view results.
Download the results as a CSV file.


# Educational Resources:

Review guidelines for spotting fake news.
Access external fact-checking resources.
Learn about the tool’s methodology and limitations.



# Project Structure

fake-news-detection-system/
├── app.py                   # Main Streamlit application script
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── data/                    # (Optional) Directory for storing datasets

# Limitations

* Model Accuracy: The logistic regression model is trained on a small example dataset, which may limit its generalization to diverse news articles. For better performance, train on a larger, real-world dataset.

* Word Cloud Dependency: Word cloud visualizations require the wordcloud package, which may not be installed by default.

* NLTK Resources: Requires internet access to download NLTK resources during setup.

* Batch Processing: Large datasets may slow down processing due to sequential text preprocessing.

* Static Model: The model is trained at runtime using the example dataset. For production, consider pre-training and saving the model with joblib.

# Future Improvements

* **Enhanced Model:** Use more advanced models (e.g., BERT, LSTM) for better accuracy.

* **Pre-trained Model:** Save and load a pre-trained model to avoid runtime training.

* **Real-time Data:** Integrate APIs to fetch news articles for analysis.

* **Multilingual Support:** Extend preprocessing and modeling to support non-English languages.

* **Performance Optimization:** Parallelize batch processing for faster analysis.

* **Accessibility:** Add support for screen readers and improve UI contrast.

# Contributing

* Contributions are welcome! To contribute:

* Fork the repository.

* Create a new branch (git checkout -b feature/your-feature).

* Commit your changes (git commit -m "Add your feature").

* Push to the branch (git push origin feature/your-feature).

* Open a pull request.



Fake News Detection System • Powered by Machine Learning • For educational purposes only


# Note: 

In datasets folder, in data file i provided the link where you can download the csv file since it was too large in size to be uploaded in github .
