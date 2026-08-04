# 🎭 Product Review Sentiment Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![NLTK](https://img.shields.io/badge/NLTK-3.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-purple)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

An end-to-end NLP project that analyzes 568,454 Amazon
food product reviews using VADER and TextBlob sentiment
analyzers, extracts distinctive keywords using TF-IDF,
and presents findings in an interactive Plotly dashboard
and a live Streamlit web application.

---

## 🎯 Business Problem

Companies receive thousands of product reviews daily.
Reading them manually is impossible at scale.
This project automatically:

- Classifies each review as POSITIVE, NEGATIVE, or NEUTRAL
- Identifies which products receive the most complaints
- Tracks how sentiment changes year over year
- Extracts the specific phrases that define each sentiment
- Flags the known limitation: VADER misclassifies 50% of
  1-star reviews due to expectation-framing language

---

## 💡 Key Findings

| Finding | Detail |
|---------|--------|
| Overall positive rate | 88.2% of reviews |
| VADER accuracy vs star ratings | 79.6% |
| 50% of 1-star reviews were classified as POSITIVE by VADER | 50% (documented limitation) |
| Most negative product | B00002N8SM at 63.2% negative |
| Best sentiment year | 2007 |
| Worst sentiment year | 2011 |
| Negative reviews length | Shorter than positive by 8 words |
| Top negative phrase | "waste money", "expiration date" |
| Top positive phrase | "highly recommend", "gluten free" |

### Critical Finding — VADER Limitation

VADER misclassifies 50% of 1-star reviews as POSITIVE.
Root cause: negative reviewers describe positive
expectations before revealing negative outcomes.

> "This was supposed to be amazing... arrived completely broken."

VADER scores "amazing" as positive and the overall
review gets a positive label despite the 1-star rating.

This highlights an important limitation of lexicon-based sentiment models when reviews contain mixed emotions or expectation-framing language.

**Production recommendation:** Combine VADER scores with
explicit star rating signals rather than relying on
text sentiment alone.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data manipulation |
| NLTK | Tokenization, stopwords |
| VADER | Rule-based sentiment scoring |
| TextBlob | Polarity + subjectivity scoring |
| Scikit-learn | TF-IDF phrase extraction |
| WordCloud | Visual word frequency maps |
| Matplotlib/Seaborn | Static charts |
| Plotly | Interactive HTML dashboard |
| Streamlit | Live web application |
| Git/GitHub | Version control |

---

## 📁 Project Structure

```text
product-sentiment-dashboard/
│
├── data/
│ ├── raw/
│ │ └── reviews.csv ← download from Kaggle
│ └── processed/
│ └── reviews_with_sentiment.csv
│
├── notebooks/
│ └── 01_sentiment_analysis.ipynb ← complete analysis
│
├── src/
│ ├── __init__.py
│ └── preprocess.py ← cleaning functions
│
├── app/
│ └── app.py ← Streamlit web app
│
├── dashboard/
│ └── sentiment_dashboard.html ← interactive dashboard
│
├── outputs/
│ ├── 01_sentiment_vs_stars_heatmap.png
│ ├── 02_confusion_matrix.png
│ ├── 03_worst_products.png
│ ├── 04_sentiment_trend_over_time.png
│ ├── 05_review_length_vs_sentiment.png
│ ├── 06_wordclouds_combined.png
│ ├── 07_tfidf_keywords.png
│ ├── wordcloud_positive.png
│ ├── wordcloud_negative.png
│ └── wordcloud_neutral.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Venkatavishnuvardhanthota/product-sentiment-dashboard.git
cd product-sentiment-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from Kaggle:
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

Save as: `data/raw/reviews.csv`

### 4. Run the notebook
Open `notebooks/01_sentiment_analysis.ipynb` in VS Code
and run all cells.

### 5. Run the Streamlit app
```bash
streamlit run app/app.py
```

### 6. View the interactive dashboard
Open `dashboard/sentiment_dashboard.html`
in any browser. No Python required.

---

## 📊 Analysis Workflow

568,454 Amazon Food Reviews (1999-2012)
↓
Text Preprocessing
→ clean_for_vader() (preserves CAPS, !!!)
→ clean_for_wordcloud() (words only)
↓
Sentiment Analysis
→ VADER compound scores
→ TextBlob polarity + subjectivity
→ 79.6% accuracy vs star ratings
↓
Exploratory Analysis
→ Sentiment vs star rating heatmap
→ Product-level negative rate ranking
→ Year-over-year sentiment trend
→ Review length by sentiment
↓
Keyword Extraction
→ WordCloud per sentiment category
→ TF-IDF distinctive phrase extraction
→ Domain stopwords removed for clean results
↓
Interactive Dashboard (Plotly HTML)

Live Web App (Streamlit)

---

## 🌐 Live Demo

**Streamlit App:** https://prduct-sentiment-dashboard.streamlit.app/

Enter any product review → instant sentiment analysis
with compound score, subjectivity, confidence level,
and key theme extraction.

---

## 📈 Charts Generated

1. Sentiment vs Star Rating Heatmap
2. Confusion Matrix — full prediction breakdown
3. Top 10 Most Negative Products
4. Sentiment Trend 1999-2012
5. Review Length by Sentiment
6. Word Clouds — Positive, Negative, Neutral
7. TF-IDF Distinctive Phrases by Sentiment

---

## 👤 Author

**Thota Venkata Vishnu Vardhan**
- Email : venkatavishnuvardhanthota@gmail.com
- GitHub: [@Venkatavishnuvardhanthota](https://github.com/Venkatavishnuvardhanthota)
- LinkedIn: [venkata-vishnu-vardhan-thota](https://linkedin.com/in/venkata-vishnu-vardhan-thota)

---

## 📄 License

Open source under the MIT License.