import pandas as pd 
import numpy as np
from collections import Counter
import string
import os
import torch
from torch.utils.data import DataLoader


class Vocabulary(object):
    """docstring for Vocabulary."""
    def __init__(self, token_to_idx=None, add_unk=True, unk_token='<UNK>'):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        # print(f'len of token_to_idx is {len(token_to_idx)}')
        
        self._idx_to_token = {idx:token for token, idx in self._token_to_idx.items()}

        self._unk_token = unk_token
        self._add_unk = add_unk
        
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def add_token(self, token):
        """Update mapping dicuts based on the token

         Args:
             token (str): the item to add to add the Vocab
         Returns:
             index (int): index corresponding to the item in the vocab dictionary
        """        
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index 
    
    def to_serializable(self):
        """ Returns a dictionary that can be serialized
        """
        return {
            'token_to_idx': self._token_to_idx,
            'add_unk': self._add_unk,
            'unk_token': self._unk_token
        }
    
    @classmethod
    def from_serializable(cls, contents):
        """Instantiates the Vocubulary from a serialized dictionary

        Args:
            contents (dict): a dictionary to create an instance of Vocabulary
        Returns: Instance of the Vocabulary class
        """

        return cls(**contents)

    def add_many(self, tokens):
        """Add a list of tokens to the vocab

        Args:
            tokens (list): List of tokens to be added to the vocab
        Returns:
            indices (list): Returns a list of indices for the tokens added
        """
        return [self.add_token(token) for token in tokens]
    
    def lookup_token(self, token):
        """Retrieve the index associated with the token 
          or the UNK index if token isn't present.
        
        Args:
            token (str): the token to look up 
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary) 
              for the UNK functionality 
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            if self.unk_index >= 0:
                index = self.unk_index
            else:
                index = -99
                print(f'{token} not present in the vocab and we dont have unk token as well')
        return index 

    def lookup_index(self, index):
        """Return the token associated with the index
        
        Args: 
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)

class ReviewVectorizer(object):
    """docstring review_ ReviewVectorizer."""
    def __init__(self, review_vocab, rating_vocab):
        """

        Args:
            review_vocab (Vocabulary): maps words to integers
            rating_vocab (Vocabulary): maps ratings to integers            
        """
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab
    
    def vectorize(self, review):
        """Create a collapsed one-hot code vector fo the review

        Args:
            review (str): the review in the str format
        Returns:
            one_hot (np.ndarray): collapsed one-hot encoding
        """

        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)

        for token in review.split(" "):
            idx = self.review_vocab.lookup_token(token) # get index for the token from the vocab class
            one_hot[idx] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        """
        Instantiate the vectorizer from the review pandas dataframe

        :param review_df (pandas df): the review dataset
        :param cutoff (int): the parameter that controls the threshold for the token to be added to the vocab
        :return:
        """

        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        # Add ratings
        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)

        # Add words that meet the cutoff to the review vocab
        word_counter = Counter()
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counter[word] += 1

        for word, count in word_counter.items():
            if count >= cutoff:
                review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab)

    def to_serializable(self):
        """
        Create the serializable dictionary for caching
        :return:
            contents (dict): the contents of the class in the form of a dictionary
        """

        return {
            'review_vocab': self.review_vocab.to_serializable(),
            'rating_vocab': self.rating_vocab.to_serializable()
        }

    @classmethod
    def from_serializable(cls, contents):
        """Instantiate a ReviewVectorizer from a serializable dictionary

        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the ReviewVectorizer class
        """

        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])

        return cls(review_vocab, rating_vocab)

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}

def update_train_state(args, model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred)>0.5).cpu().long()#.max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)