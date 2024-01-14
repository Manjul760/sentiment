import random 
import time
import json
from os import path
current_dir = path.dirname(path.realpath(__file__))
random.seed(time.time())
import re
import random
import pandas as pd
from collections import defaultdict

def preprocess_text(text):
    # Convert to lowercase and remove non-alphabetic characters
    if isinstance(text, str):
        text = re.sub(r'[^a-z\s]', '', text.lower())
    return text

def extract_features(document):
    # Extract word features (unigram)
    features = defaultdict(int)
    
    # Handle both string and defaultdict cases
    if isinstance(document, str):
        for word in document.split():
            features[word] += 1
    elif isinstance(document, defaultdict):
        features = document
    
    return features

def split_data(data, split_ratio=0.8):
    # Split data into training and testing sets
    random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    train_set = data[:split_index]
    test_set = data[split_index:]
    return train_set, test_set

def train_naive_bayes(train_set):
    # Train Naive Bayes classifier
    positive_docs = [(doc, label) for doc, label in train_set if label == 'positive']
    negative_docs = [(doc, label) for doc, label in train_set if label == 'negative']
    neutral_docs = [(doc, label) for doc, label in train_set if label == 'neutral']

    # Calculate class probabilities
    prior_positive = len(positive_docs) / len(train_set)
    prior_negative = len(negative_docs) / len(train_set)
    prior_neutral = len(neutral_docs) / len(train_set)

    # Calculate word probabilities given the class
    positive_word_probs = defaultdict(float)
    negative_word_probs = defaultdict(float)
    neutral_word_probs = defaultdict(float)

    for doc, _ in positive_docs:
        for word, count in doc.items():
            positive_word_probs[word] += count / len(positive_docs)

    for doc, _ in negative_docs:
        for word, count in doc.items():
            negative_word_probs[word] += count / len(negative_docs)

    for doc, _ in neutral_docs:
        for word, count in doc.items():
            neutral_word_probs[word] += count / len(neutral_docs)

    return {
        "prior_positive": prior_positive, 
        "prior_negative": prior_negative, 
        "prior_neutral":  prior_neutral, 
        "positive_word_probs": positive_word_probs, 
        "negative_word_probs": negative_word_probs, 
        "neutral_word_probs": neutral_word_probs
    }

def classify_naive_bayes(document, prob_dict):
    # Classify document using Naive Bayes
    positive_score = prob_dict["prior_positive"]
    negative_score = prob_dict["prior_negative"]
    neutral_score = prob_dict["prior_neutral"]

    for word, count in document.items():
        try:
            positive_score *= prob_dict["positive_word_probs"][word] ** count
            negative_score *= prob_dict["negative_word_probs"][word] ** count
            neutral_score *= prob_dict["neutral_word_probs"][word] ** count
        except:continue

    # Choose the class with the highest score
    scores = {'positive': positive_score, 'negative': negative_score, 'neutral': neutral_score}
    predicted_label = max(scores, key=scores.get)

    return predicted_label

if __name__=="__main__":
    positive_reviews = ["I love this product. It's amazing!", "Great service, highly recommend."]
    negative_reviews = ["Terrible experience. Waste of money.", "Poor quality, do not buy."]
    neutral_reviews = ["The product is okay.", "No strong opinion."]


    data  = pd.read_csv(path.join(current_dir,"test.csv"),encoding='ISO-8859-1',usecols=["text","sentiment"])
    for i,row in data.iterrows():
        text = row['text']
        sentiment = row['sentiment']

        if sentiment == 'positive': positive_reviews.append(text)
        elif sentiment == 'negative': negative_reviews.append(text)
        elif sentiment == 'neutral': neutral_reviews.append(text)



    positive_feature_set = [(extract_features(preprocess_text(review)), 'positive') for review in positive_reviews]
    negative_feature_set = [(extract_features(preprocess_text(review)), 'negative') for review in negative_reviews]
    neutral_feature_set = [(extract_features(preprocess_text(review)), 'neutral') for review in neutral_reviews]

    all_feature_set = positive_feature_set + negative_feature_set + neutral_feature_set


    train_set, test_set = split_data(all_feature_set)
    prob_dict = train_naive_bayes(train_set)

    with open(path.join(current_dir,"Sentiments.json"),"w") as f: json.dump(prob_dict,f)

    # Test the classifier
    correct_predictions = 0
    for doc, label in test_set:
        features = extract_features(preprocess_text(doc))
        predicted_label = classify_naive_bayes(features, prob_dict)
        if predicted_label == label: correct_predictions += 1

    accuracy = correct_predictions / len(test_set)
    print(f"Accuracy: {accuracy}")




