# import transformers
# import tokenizers
# pylint: disable=no-member

import os
from argparse import Namespace
from model import ReviewMLPClassifier, ReviewPerceptronClassifier, ReviewMLPEmbClassifier
from dataset import ReviewDataset
from utils import handle_dirs

args = Namespace(
    # Data and Path information
    frequency_cutoff=25,
    model_state_file='model_2.pth',
    review_csv='../input/reviews_with_splits_lite.csv',
    # review_csv='data/yelp/reviews_with_splits_full.csv',
    save_dir='../experiment/perceptron/',
    vectorizer_file='vectorizer.json',
    classifier=None,
    vectorizer = None,
    dataset=None,
    # No Model hyper parameters
    # Training hyper parameters
    batch_size=32,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=2,
    seed=1337,
    # Runtime options
    catch_keyboard_interrupt=True,
    cuda=False,
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
    train=True,
    emb=True
)
# handle dirs
handle_dirs(args.save_dir)

vectorizer_pth = os.path.join(args.save_dir, args.vectorizer_file)
if args.reload_from_files:
        # training from a checkpoint
        print("Loading dataset and vectorizer")        
        dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv,
                                                                 vectorizer_pth, args.emb)
else:
        print("Loading dataset and creating vectorizer")
        # create dataset and vectorizer
        dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv, args.emb)
        dataset.save_vectorizer(vectorizer_pth)

vectorizer = dataset.get_vectorizer()

if args.emb:
    embedded_cols = {f'col_{i}': 7497 for i in range(10)}
    embedding_sizes = [(n_categories, min(500, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]
    classifier = ReviewMLPEmbClassifier(embedding_sizes, num_features=10, num_classes=1)
else:
    classifier = ReviewPerceptronClassifier(num_features=len(vectorizer.review_vocab), num_classes=1)
# classifier = ReviewMLPClassifier(num_features=len(vectorizer.review_vocab), num_classes=1, hidden_layer_dim=[100])


args.classifier = classifier
args.vectorizer = vectorizer
args.dataset = dataset

if __name__== '__main__':
    print(args)

    