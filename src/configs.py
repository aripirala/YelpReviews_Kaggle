# import transformers
# import tokenizers
# pylint: disable=no-member

import os
from argparse import Namespace
from model import ReviewMLPClassifier, ReviewPerceptronClassifier, ReviewMLP_Embed_Classifier, ReviewCNN_Embed_Classifier, ReviewRNN_Embed_Classifier
from dataset import ReviewDataset
from torch.nn.modules.dropout import Dropout
from utils import handle_dirs

args = Namespace(
    # Data and Path information
    frequency_cutoff=25,
    model_state_file='model_embed_gru_len200.pth',
    review_csv='../input/reviews_with_splits_lite.csv',
    # review_csv='data/yelp/reviews_with_splits_full.csv',
    save_dir='../experiment/embedding_rnn/',
    vectorizer_file='vectorizer.json',
    classifier=None,
    vectorizer = None,
    dataset=None,
    architecture_type=None,
    # No Model hyper parameters
    # Training hyper parameters
    batch_size=32,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=10,
    seed=1337,
    # Runtime options
    catch_keyboard_interrupt=True,
    cuda=True,
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
    train=True, # Flag to train your network
    # If embedding layer is used
    max_len = 200,
    vector_type='embedding',
    embedding_type = 'pre-trained',
    embedding_file_name= '../input/glove.6B.50d.txt',
    embedding_dim=50
)
# handle dirs
handle_dirs(args.save_dir)

vectorizer_pth = os.path.join(args.save_dir, args.vectorizer_file)
if args.reload_from_files:
        # training from a checkpoint
        print("Loading dataset and vectorizer")        
        dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv,
                                                                 vectorizer_pth)
else:
        print("Loading dataset and creating vectorizer")
        # create dataset and vectorizer
        dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv, 
                                                                args.vector_type, 
                                                                args.max_len)
        dataset.save_vectorizer(vectorizer_pth)

vectorizer = dataset.get_vectorizer()


# classifier = ReviewPerceptronClassifier(num_features=len(vectorizer.review_vocab), num_classes=1)
# classifier = ReviewMLPClassifier(num_features=len(vectorizer.review_vocab), num_classes=1, hidden_layer_dim=[100])
# classifier = ReviewMLP_Embed_Classifier(num_features=len(vectorizer.review_vocab), num_classes=1, hidden_layer_dim=[100, 50],
#                 embedding_file_name=args.embedding_file_name, embedding_dim=50,  
#                 word_to_index=vectorizer.review_vocab._token_to_idx, max_idx=len(vectorizer.review_vocab),
#                 freeze=False)
# args.architecture_type = 'MLP'

# classifier = ReviewCNN_Embed_Classifier(num_features=len(vectorizer.review_vocab), num_classes=1, channel_list=[100, 200, 400],
#                 embedding_file_name=args.embedding_file_name, embedding_dim=args.embedding_dim,  
#                 word_to_index=vectorizer.review_vocab._token_to_idx, max_idx=len(vectorizer.review_vocab),
#                 freeze=False, batch_norm=True, dropout=True, max_pool=True, activation_fn='ELU')
# args.architecture_type = 'CNN'

classifier = ReviewRNN_Embed_Classifier(num_features=len(vectorizer.review_vocab), num_classes=1, rnn_hidden_size=200,
                embedding_file_name=args.embedding_file_name, embedding_dim=args.embedding_dim,  
                word_to_index=vectorizer.review_vocab._token_to_idx, max_idx=len(vectorizer.review_vocab),
                freeze=True, batch_norm=True, dropout=True, activation_fn='RELU')
args.architecture_type = 'RNN'


args.classifier = classifier
args.vectorizer = vectorizer
args.dataset = dataset

if __name__== '__main__':
    # print(args)
    review_dataset = args.dataset
    # print(args.dataset._lookup_dict)
    train_dataset = review_dataset 
    print(f'Training dataset has {len(train_dataset)}')
    print('First five items are --')
    for i in range(5):
        x, y = train_dataset[i]['x_data'], train_dataset[i]['y_target']
        print(f'data {i+1}...\n\t{x}\n\t{y}')

    review_dataset.set_split('val')
    print(f'Validation dataset has {len(review_dataset)}')
    print('Two items are --')
    for i in range(2):
        x, y = review_dataset[i]['x_data'], review_dataset[i]['y_target']
        print(f'data {i+1}...\n\t{x}\n\t{y}')

    review_dataset.set_split('train')
    print(f'Training dataset has {len(review_dataset)}')

    review_dataset.set_split('test')
    print(f'Test dataset has {len(review_dataset)}')

    