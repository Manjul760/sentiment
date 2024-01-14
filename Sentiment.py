from SentimentTrain import *

try:
    prob_dict={}
    with open(path.join(current_dir,"Sentiments.json"),"r") as f: prob_dict=json.load(f)
    with open(path.join(current_dir,"SentimentTemp.txt"),"r") as f:
        new_review=f.read()
        new_features = extract_features(preprocess_text(new_review))
        predicted_sentiment = classify_naive_bayes(new_features, prob_dict)
        print(f"{predicted_sentiment}")
except:
    print("error analysing")

















