# pylint: disable=no-member
# pylint: disable=E1101
# pylint: disable=E1102

import torch
from utils import preprocess_text

from model import ReviewPerceptronClassifier
from utils import ReviewVectorizer
from configs import args
import os

import torch.nn.functional as F


def predict_rating(review, classifier, vectorizer, decision_threshold=0.5):
    """Predict the rating of a review

    Args:
        review (str): the text of the review
        classifier (ReviewClassifier): the trained model
        vectorizer (ReviewVectorizer): the corresponding vectorizer
        decision_threshold (float): The numerical boundary which separates the rating classes
    """
    review = preprocess_text(review)

    vectorized_review = torch.tensor(vectorizer.vectorize(review))
    print(vectorized_review)
    result = classifier(vectorized_review.view(1, -1))

    probability_value = torch.sigmoid(result).item()
    index = 1
    if probability_value < decision_threshold:
        index = 0

    return vectorizer.rating_vocab.lookup_index(index), probability_value

if __name__ == '__main__':
    test_review = "this is not so bad. There were some good dishes"

    #get the model and vectorizer paths
    model_pth = os.path.join(args.save_dir, args.model_state_file)
    vectorizer_pth = os.path.join(args.save_dir, args.vectorizer_file)

    #load vectorizer before loading the model
    vectorizer = ReviewVectorizer.from_serializable_and_json(vectorizer_pth)
    print(f'Length of review vocab is {len(vectorizer.review_vocab)}')

    classifier = ReviewPerceptronClassifier(num_features=len(vectorizer.review_vocab), num_classes=1)
    classifier.load_state_dict(torch.load(model_pth))
    classifier = classifier.cpu()

    prediction, probability = predict_rating(test_review, classifier, vectorizer, decision_threshold=0.5)
    print(f"{test_review} -> {prediction} with a probability of {probability}")