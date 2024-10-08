from textblob import TextBlob
from datasets import load_dataset

####detecting emotion function
def get_sentiment(text):
    blob = TextBlob(text)
    
    sentiment_polarity = blob.sentiment.polarity
    if sentiment_polarity > 0:
        return "Positive"
    elif sentiment_polarity < 0:
        return "Negative"
    else:
        return "Neutral"

dataset = load_dataset('go_emotions', split='train')
texts = dataset['text']


for text in texts:
    sentiment = get_sentiment(text)
    print(f"Text: {text}\nSentiment: {sentiment}\n")
