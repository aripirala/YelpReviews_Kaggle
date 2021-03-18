# pylint: disable=no-member

import torch
import torch.nn as nn
import sys
import timeit
import datetime
import numpy as np
import sys



class ReviewPerceptronClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ReviewPerceptronClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, num_classes)

    def forward(self, x_in, apply_sigmoid=False):
        logits = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            logits = torch.sigmoid(logits)

        return logits

class ReviewMLPClassifier(nn.Module):
    def __init__(self, num_features, num_classes, hidden_layer_dim=None, activation_fn = 'RELU'):
        super(ReviewMLPClassifier, self).__init__()
        # self.fc1 = nn.Linear(num_features, num_classes)
        layers = []
        input_features = num_features
        if hidden_layer_dim is not None:
            for hidden_features in hidden_layer_dim:
                layers.append(nn.Linear(input_features, hidden_features))
                if activation_fn == 'RELU':
                    layers.append(nn.ReLU())
                input_features = hidden_features
        layers.append(nn.Linear(input_features, num_classes))        
        self.model = nn.Sequential(*layers)

    def forward(self, x_in, apply_sigmoid=False):
        logits = self.model(x_in).squeeze()
        if apply_sigmoid:
            logits = torch.sigmoid(logits)
        return logits

class ReviewMLP_Embed_Classifier(nn.Module):
    def __init__(self, num_features, num_classes, hidden_layer_dim=None, activation_fn = 'RELU', \
                    embedding_dim=100, embedding_type='pre-trained', embedding_file_name=None, word_to_index=None, max_idx=1000, freeze=True):
        super(ReviewMLP_Embed_Classifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        if embedding_type=='pre-trained':
            self.emb_matrix = self.create_embedding_matrix(embedding_file_name, word_to_index, max_idx)
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.emb_matrix), freeze=freeze)
            print(f'Embeddings loaded - shape is {self.emb_matrix.shape}')
            # sys.exit()
        else:
            self.embedding = nn.Embedding(num_features, embedding_dim)
        layers = []
        
        input_features = embedding_dim

        if hidden_layer_dim is not None:
            for hidden_features in hidden_layer_dim:
                layers.append(nn.Linear(input_features, hidden_features))
                if activation_fn == 'RELU':
                    layers.append(nn.ReLU())
                    layers.append(nn.BatchNorm1d(hidden_features))
                    layers.append(nn.Dropout(p=0.25))
                    
                input_features = hidden_features
        layers.append(nn.Linear(input_features, num_classes))        
        self.seq_model = nn.Sequential(*layers)
        
    def create_embedding_matrix(self, embeddings_file_name, word_to_index, max_idx, sep=' ', init='zeros', print_each=10000, verbatim=False):
        # Initialize embeddings matrix to handle unknown words
        if init == 'zeros':
            embed_mat = np.zeros((max_idx + 1, self.embedding_dim))
        elif init == 'random':
            embed_mat = np.random.rand(max_idx + 1, self.embedding_dim)
        else:
            raise Exception('Unknown method to initialize embeddings matrix')
        
        start = timeit.default_timer()
        with open(embeddings_file_name) as infile:
            for idx, line in enumerate(infile):
                elem = line.split(sep)
                word = elem[0]

                if verbatim is True:
                    if idx % print_each == 0:
                        print('[{}] {} lines processed'.format(datetime.timedelta(seconds=int(timeit.default_timer() - start)), idx), end='\r')

                if word not in word_to_index:
                    continue

                word_idx = word_to_index[word]

                if word_idx <= max_idx:
                    embed_mat[word_idx] = np.asarray(elem[1:], dtype='float32')


        if verbatim == True:
            print()

        return embed_mat

    def forward(self, x_in, apply_sigmoid=False):
        # print(f'x_input shape is {x_in.size()}')
        x_in = x_in.to(torch.long)
        embed_out = self.embedding(x_in)
        embed_out = embed_out.permute(0, 2, 1)
        # print(f'embedding shape is {embed_out.size()}')
        embed_out,_ = torch.max(embed_out, dim=2)
        # print(f'embedding shape is {embed_out.size()}')
        
        # sys.exit()

        logits = self.seq_model(embed_out.float()).squeeze()
        if apply_sigmoid:
            logits = torch.sigmoid(logits)
        return logits

if __name__ == '__main__':
    hidden_layer_dim = [50, 25]
    mlpClassifier = ReviewMLP_Embed_Classifier(200, 1, hidden_layer_dim)
    print(mlpClassifier)
