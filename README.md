# Tweet-Sentiment-Analyzer
Tweet Sentiment Analyzer is a Python-based application that utilizes machine learning to determine the sentiment of tweets. It processes textual data to classify sentiments as positive, negative, or neutral, providing valuable insights for social media analysis, customer feedback, and trend identification. Built with Python, it leverages powerful libraries like scikit-learn and NLTK for natural language processing and model training.

# Tweet Sentiment Analyzer

This repository contains a **Tweet Sentiment Analyzer**, a machine learning project aimed at analyzing sentiments (positive or negative) in tweets using **Natural Language Processing (NLP)** techniques and a **Logistic Regression Model**.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Workflow](#project-workflow)
4. [Requirements](#requirements)
5. [Installation and Usage](#installation-and-usage)
6. [Model Training](#model-training)
7. [Results](#results)
8. [Future Scope](#future-scope)

---

## Project Overview

The **Tweet Sentiment Analyzer** processes a dataset of tweets to classify each tweet as either:
- **Positive Sentiment**
- **Negative Sentiment**

Key tasks include:
- Data cleaning and preprocessing.
- Text vectorization using **TF-IDF**.
- Training a **Logistic Regression Model**.
- Evaluating the model's accuracy.

---

## Dataset

> Dataset Source: Kaggle

> The project utilizes a dataset containing **1.6 million tweets**, each labeled as either:
- `0`: Negative sentiment.
- `4` (replaced with `1`): Positive sentiment.

> Dataset Columns:
1. `target`: Sentiment label (`0` or `1`).
2. `id`: Unique identifier for each tweet.
3. `date`: Date of the tweet.
4. `flag`: Query flag (unused in analysis).
5. `user`: Username of the tweet author.
6. `text`: Actual tweet content.

---

## Project Workflow

1. **Import Libraries**: Utilizes `numpy`, `pandas`, `nltk`, `sklearn`, and more.
2. **Data Preprocessing**:
   - Remove special characters, links, and mentions.
   - Convert text to lowercase.
   - Remove stopwords using NLTK's stopwords library.
   - Apply stemming using the **Porter Stemmer**.
3. **Vectorization**: Use **TF-IDF Vectorizer** to transform text into numerical features.
4. **Model Training**: Train a Logistic Regression model on preprocessed data.
5. **Model Evaluation**: Assess accuracy on the test set.

---

## Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `nltk`
  - `scikit-learn`

---

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/tweet-sentiment-analyzer.git
   cd tweet-sentiment-analyzer
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and place it in the project directory. Use the following naming convention for the dataset file: **data.csv**

4. Run the script:

   ```bash
   python sentiment_analyzer.py
   ```

---

## Model Training

- **Training Data**: 80% of the dataset.
- **Test Data**: 20% of the dataset.
- **Vectorization**: Features are extracted using TF-IDF.
  
---

## Results

- The Logistic Regression model achieved:
  - Training Accuracy: ~85%
  - Testing Accuracy: ~84%
- Sentiment prediction examples:
  - Input: I love this!
  - Output: Positive
  - Input: This is terrible.
  - Output: Negative


---

## Future Scope

1. Enhance preprocessing to handle sarcasm and emojis.
2. Experiment with advanced machine learning models like Random Forests or Neural Networks.
3. Deploy the model using a web interface or REST API.
4. Integrate multilingual support for analyzing non-English tweets.

