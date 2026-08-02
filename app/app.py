# Customer Review Sentiment Analyzer
# Streamlit Web Application

import streamlit as st
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.preprocess import (
    clean_for_vader,
    clean_for_wordcloud
)

#  Page Configuration 

st.set_page_config(
    page_title="Review Sentiment Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  Load NLP Tools 

@st.cache_resource
def load_nlp_tools():
    from vaderSentiment.vaderSentiment import (
        SentimentIntensityAnalyzer
    )
    import nltk
    from nltk.corpus import stopwords

    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    analyzer = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words('english'))

    return analyzer, stop_words


analyzer, STOP_WORDS = load_nlp_tools()

#  Helper Functions 

def get_sentiment_label(compound):

    if compound >= 0.05:
        return 'POSITIVE'
    elif compound <= -0.05:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'


def get_confidence(compound):
    return abs(compound) * 100


def extract_keywords(text, top_n=8):

    if not text or text.strip() == '':
        return []

    words = text.split()

    word_freq = {}

    for word in words:
        if len(word) > 2:
            word_freq[word] = word_freq.get(word, 0) + 1

    sorted_words = sorted(
        word_freq.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        word
        for word, count in sorted_words[:top_n]
    ]


#  App Header 

st.title("🔍 Product Review Sentiment Analyzer")

st.markdown(
    "Analyze any product review using "
    "**VADER** and **TextBlob** NLP engines. "
    "Instantly detect sentiment and key themes."
)

st.divider()

#  Sidebar 

with st.sidebar:

    st.header("ℹ️ About This App")

    st.markdown("""
    **How it works:**
    1. Type or paste any product review
    2. Click **Analyze Review**
    3. See instant sentiment analysis!

    **Powered by:**
    - VADER Sentiment Analyzer
    - TextBlob NLP Library
    - Amazon Food Reviews Dataset
      (568,454 reviews analyzed)

    **Accuracy:** 79.6% vs human star ratings
    """)

    st.divider()

    st.header("📊 Dataset Insights")

    st.metric(
        "Total Reviews Analyzed",
        "568,454"
    )

    st.metric(
        "Time Period",
        "1999 — 2012"
    )

    st.metric(
        "VADER Accuracy",
        "79.6%"
    )

    st.divider()

    st.header("🧪 Try These Examples")

    example_reviews = {
        "Positive Example": (
            "Absolutely love this product! "
            "Best coffee I have ever tasted. "
            "Will definitely order again!!!"
        ),
        "Negative Example": (
            "Complete waste of money. "
            "Arrived expired and tasted terrible. "
            "Requested a refund immediately."
        ),
        "Neutral Example": (
            "The product is okay. "
            "Nothing special about it. "
            "Packaging was decent."
        ),
        "Tricky Example": (
            "I expected this to be amazing "
            "based on the reviews but it was "
            "a complete disappointment."
        )
    }

    for label, review_text in example_reviews.items():

        if st.button(
            label,
            use_container_width=True
        ):
            st.session_state['review_input'] = review_text

#  Main Input Area 

col1, col2 = st.columns([2, 1])

with col1:

    st.subheader("📝 Enter Your Review")

    review_input = st.text_area(
        label="Review Text",
        value=st.session_state.get(
            'review_input',
            ''
        ),
        height=150,
        placeholder=(
            "Paste or type any product review here...\n"
            "Example: This coffee is absolutely amazing! "
            "Best purchase I've made this year."
        ),
        key="review_text_area"
    )

    analyze_button = st.button(
        "🔍 Analyze Review",
        type="primary",
        use_container_width=True
    )

with col2:

    st.subheader("📈 Dataset Benchmark")

    st.markdown("""
    From our analysis of 568,454 reviews:

    | Sentiment | % of Reviews |
    |-----------|-------------|
    | 😊 Positive | 88.2% |
    | 😠 Negative | 9.8% |
    | 😐 Neutral | 2.0% |

    *Your review will be compared against
    these patterns.*
    """)

#  Analysis Results 

if analyze_button and review_input.strip():

    st.divider()
    st.subheader("📊 Analysis Results")

    cleaned_vader = clean_for_vader(
        review_input
    )

    cleaned_wc = clean_for_wordcloud(
        review_input
    )

    vader_scores = analyzer.polarity_scores(
        cleaned_vader
    )

    compound = vader_scores['compound']

    sentiment_label = get_sentiment_label(
        compound
    )

    confidence = get_confidence(
        compound
    )

    from textblob import TextBlob

    blob = TextBlob(cleaned_vader)

    tb_polarity = blob.sentiment.polarity
    tb_subjectivity = blob.sentiment.subjectivity

    keywords = extract_keywords(cleaned_wc)

    sentiment_config = {
        'POSITIVE': {
            'color': '#2ecc71',
            'emoji': '😊',
            'bg': '#d5f5e3',
            'message': 'This review expresses positive sentiment!'
        },
        'NEGATIVE': {
            'color': '#e74c3c',
            'emoji': '😠',
            'bg': '#fadbd8',
            'message': 'This review expresses negative sentiment.'
        },
        'NEUTRAL': {
            'color': '#3498db',
            'emoji': '😐',
            'bg': '#d6eaf8',
            'message': 'This review expresses neutral sentiment.'
        }
    }

    config = sentiment_config[sentiment_label]

    st.markdown(
        f"""
        <div style="
            background-color: {config['bg']};
            border-left: 6px solid {config['color']};
            padding: 20px;
            border-radius: 8px;
            margin: 10px 0;
        ">
            <h2 style="color: {config['color']};
                        margin: 0;">
                {config['emoji']} {sentiment_label}
            </h2>
            <p style="margin: 5px 0; font-size: 16px;">
                {config['message']}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            label="VADER Compound",
            value=f"{compound:+.3f}",
            help="Overall sentiment score: -1 to +1"
        )

    with m2:
        st.metric(
            label="Confidence",
            value=f"{confidence:.1f}%",
            help="Higher = more confident"
        )

    with m3:
        st.metric(
            label="Subjectivity",
            value=f"{tb_subjectivity:.3f}",
            help="0 = objective, 1 = opinion"
        )

    with m4:
        st.metric(
            label="TextBlob Polarity",
            value=f"{tb_polarity:+.3f}",
            help="TextBlob polarity score"
        )

    st.subheader("🔢 VADER Score Breakdown")

    score_col1, score_col2 = st.columns(2)

    with score_col1:

        scores_data = {
            'Component': [
                'Positive',
                'Negative',
                'Neutral',
                'Compound'
            ],
            'Score': [
                vader_scores['pos'],
                vader_scores['neg'],
                vader_scores['neu'],
                vader_scores['compound']
            ]
        }

        import pandas as pd

        st.dataframe(
            pd.DataFrame(scores_data),
            hide_index=True,
            use_container_width=True
        )

    with score_col2:
        st.markdown(
            "<p style='font-size:28px;'><b>Sentiment Strength</b></p>",
            unsafe_allow_html=True
                    )

        normalized = (compound + 1) / 2 * 100

        scale_col1, scale_col2, scale_col3 = st.columns(3)

        with scale_col1:
            st.markdown(
                "<p style='font-size:12px; "
                "color:#e74c3c; margin:0'>◄ Very Negative</p>",
                unsafe_allow_html=True
            )

        with scale_col2:
            st.markdown(
                "<p style='font-size:12px; "
                "text-align:center; margin:0'>Neutral</p>",
                unsafe_allow_html=True
            )

        with scale_col3:
            st.markdown(
                "<p style='font-size:12px; "
                "color:#2ecc71; "
                "text-align:right; margin:0'>"
                "Very Positive ►</p>",
                unsafe_allow_html=True
            )

        bar_color = config['color']

        st.markdown(
            f"""
            <div style="
                background: #f0f0f0;
                border-radius: 10px;
                height: 24px;
                width: 100%;
                position: relative;
                margin: 4px 0;
            ">
                <div style="
                    background: {bar_color};
                    width: {normalized:.1f}%;
                    height: 100%;
                    border-radius: 10px;
                    transition: width 0.3s;
                "></div>
                <div style="
                    position: absolute;
                    left: 50%;
                    top: 0;
                    height: 100%;
                    width: 2px;
                    background: #666;
                "></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"<p style='text-align:center; "
            f"font-size:20px; font-weight:bold; "
            f"color:{bar_color}; margin:8px 0'>"
            f"{compound:+.4f}</p>",
            unsafe_allow_html=True
        )

        if abs(compound) >= 0.75:
            strength = "Very Strong"
        elif abs(compound) >= 0.5:
            strength = "Strong"
        elif abs(compound) >= 0.25:
            strength = "Moderate"
        elif abs(compound) >= 0.05:
            strength = "Mild"
        else:
            strength = "Neutral"

        st.markdown(
            f"<p style='text-align:center; "
            f"font-size:13px; color:#666; margin:0'>"
            f"{strength} {sentiment_label.title()}</p>",
            unsafe_allow_html=True
        )

    if keywords:

        st.subheader("🔑 Key Themes Detected")

        keyword_cols = st.columns(len(keywords))

        for col, keyword in zip(keyword_cols, keywords):

            with col:

                st.markdown(
                    f"""
                    <div style="
                        background: {config['bg']};
                        border: 1px solid {config['color']};
                        border-radius: 20px;
                        padding: 5px 10px;
                        text-align: center;
                        font-size: 13px;
                        font-weight: bold;
                        color: {config['color']};
                    ">
                        {keyword}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander(
        "🔍 See How Text Was Cleaned"
    ):

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown(
                "**For VADER (minimal cleaning):**"
            )
            st.text(cleaned_vader[:300])

        with exp_col2:
            st.markdown(
                "**For Keywords (full cleaning):**"
            )
            st.text(cleaned_wc[:300])

    st.info(
        "💡 **Note:** VADER achieves 79.6% accuracy "
        "against human star ratings on Amazon reviews. "
        "Always combine with star ratings in production."
    )

elif analyze_button and not review_input.strip():

    st.warning(
        "⚠️ Please enter a review to analyze."
    )