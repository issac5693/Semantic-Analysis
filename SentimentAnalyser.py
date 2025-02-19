from nltk import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def preprocess_text(text):
    #tokenize text 
    tokenized_text= word_tokenize(text.lower())

    #remove stop words
    filtered_tokens= [token for token in tokenized_text if token not in stopwords.words("english")]

    #lemmatize filtered token 
    lemmatized_tokens= [WordNetLemmatizer().lemmatize(fil_token) for fil_token in filtered_tokens]

    preprocessed_text= " ".join(lemmatized_tokens)
    return preprocessed_text

def get_sentiment(text):
    sentiment= SentimentIntensityAnalyzer().polarity_scores(text)
    return 1 if sentiment['pos'] > 0 else 0

preprocessed_text= preprocess_text("he's a funny guy and I like him")
print(get_sentiment(preprocessed_text))