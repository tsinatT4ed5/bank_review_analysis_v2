import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

# Download necessary NLTK resources (only needs to be done once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()
tqdm.pandas()

# --- 1. Load Data ---
try:
    df = pd.read_csv('data/clean_bank_reviews.csv')
    print(f"Loaded {len(df)} reviews for analysis.")
except FileNotFoundError:
    print("Error: clean_bank_reviews.csv not found. Please ensure Task 1 is complete.")
    exit()

# --- 2. Text Preprocessing ---

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Lowercase
    text = text.lower()
    # Tokenize and remove stop words and lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

print("Starting text preprocessing (tokenization, stop-word removal, lemmatization)...")
df['processed_review'] = df['review'].progress_apply(preprocess_text)

# --- 3. Sentiment Analysis (VADER) ---

def get_vader_sentiment(text):
    vs = analyzer.polarity_scores(text)
    # Determine the label based on the compound score
    if vs['compound'] >= 0.05:
        label = 'Positive'
    elif vs['compound'] <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
    return label, vs['compound']

print("Starting VADER sentiment analysis...")
# Apply sentiment analysis and unpack the results into two new columns
df[['sentiment_label', 'sentiment_score']] = df['processed_review'].progress_apply(
    lambda x: pd.Series(get_vader_sentiment(x))
)

# --- 4. Aggregate Sentiment Results (KPI Check) ---

# Aggregate sentiment by bank
sentiment_summary = df.groupby('bank')['sentiment_label'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
print("\n--- Sentiment Summary by Bank (%) ---")
print(sentiment_summary.round(2))

# Aggregate sentiment by rating
sentiment_by_rating = df.groupby('rating')['sentiment_label'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
print("\n--- Sentiment Summary by Rating (%) ---")
print(sentiment_by_rating.round(2))

# Check KPI: Sentiment scores for 90%+ reviews
reviews_with_sentiment = df['sentiment_score'].notna().sum()
sentiment_coverage = (reviews_with_sentiment / len(df)) * 100
print(f"\nKPI Check: Sentiment scores calculated for {sentiment_coverage:.2f}% of reviews.")

# --- 5. Thematic Analysis (Keyword Extraction & Clustering) ---

from sklearn.feature_extraction.text import TfidfVectorizer

# 5.1. Keyword Extraction (TF-IDF)
def get_top_keywords(df_bank, n=10):
    # Use the processed review text
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df_bank['processed_review'])
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate feature scores (sum of TF-IDF scores)
    scores = tfidf_matrix.sum(axis=0).A1
    
    # Create a Series of scores and sort
    feature_scores = pd.Series(scores, index=feature_names).sort_values(ascending=False)
    return feature_scores.head(n)

# 5.2. Rule-Based Thematic Clustering
# Define keywords for 5 common themes based on typical banking app reviews
THEME_KEYWORDS = {
    'Account Access & Security': ['login', 'fingerprint', 'password', 'otp', 'secure', 'access', 'locked', 'verification'],
    'Transaction Performance': ['transfer', 'send', 'receive', 'slow', 'fast', 'instant', 'delay', 'transaction', 'payment'],
    'User Interface & Experience': ['ui', 'interface', 'design', 'easy', 'confusing', 'update', 'bug', 'crash', 'smooth'],
    'Customer Support & Service': ['support', 'customer', 'call', 'branch', 'service', 'help', 'response', 'staff'],
    'Feature Requests & Reliability': ['feature', 'reliable', 'work', 'always', 'never', 'option', 'function', 'problem']
}

def assign_theme(review_text):
    assigned_themes = []
    for theme, keywords in THEME_KEYWORDS.items():
        # Check if any keyword is present in the review text
        if any(keyword in review_text for keyword in keywords):
            assigned_themes.append(theme)
    return assigned_themes if assigned_themes else ['Other']

# 5.3. Apply Thematic Analysis to each bank
print("\n--- Thematic Analysis (Top Keywords & Theme Assignment) ---")
final_themes = {}

for bank in df['bank'].unique():
    df_bank = df[df['bank'] == bank].copy()
    
    # Get top keywords
    top_keywords = get_top_keywords(df_bank, n=15)
    final_themes[bank] = top_keywords.index.tolist()
    
    print(f"\nTop 15 Keywords for {bank}:")
    print(top_keywords)
    
    # Assign themes
    df.loc[df['bank'] == bank, 'identified_themes'] = df_bank['processed_review'].apply(assign_theme)

# 5.4. Aggregate Theme Results
theme_counts = df.explode('identified_themes').groupby('bank')['identified_themes'].value_counts().unstack(fill_value=0)
print("\n--- Theme Counts by Bank ---")
print(theme_counts)

# --- 6. Final Save ---
# Select final columns for the output CSV
final_cols = ['review', 'rating', 'date', 'bank', 'source', 'sentiment_label', 'sentiment_score', 'identified_themes']
df_final = df[final_cols]

# Save the final dataframe with sentiment and themes
df_final.to_csv('data/final_analysis_results.csv', index=False)
print("\nFinal analysis results saved to data/final_analysis_results.csv")

