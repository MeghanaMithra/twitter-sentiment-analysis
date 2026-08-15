import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon (runs once automatically)
nltk.download('vader_lexicon', quiet=True)

class SentimentEngine:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def clean_tweet(self, text: str) -> str:
        """Cleans tweet text using NLP regular expressions."""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'@\w+', '', text)                                         # Remove handles
        text = re.sub(r'#(\w+)', r'\1', text)                                    # Remove '#' symbol
        text = re.sub(r'[^\w\s\d]', '', text)                                    # Remove special characters
        return text.strip()

    def analyze(self, text: str) -> dict:
        """Returns cleaned text, compound score, and sentiment label."""
        cleaned_text = self.clean_tweet(text)
        scores = self.sia.polarity_scores(cleaned_text)
        compound = scores['compound']

        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {
            'raw_text': text,
            'clean_text': cleaned_text,
            'compound': compound,
            'sentiment': sentiment
        }