# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import timeit
import datetime
import numpy as np
import sys
from utils import column_gather



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

class Embedding:
    def __init__(self, num_features, embedding_dim=100, embedding_type='pre-trained', embedding_file_name=None, 
                    word_to_index=None, max_idx=1000, freeze=True, **kwargs):
        
        super().__init__(**kwargs)
        
        self.embedding_dim= embedding_dim

        if embedding_type is not None:
            self.emb_matrix = self.create_embedding_matrix(embedding_file_name, word_to_index, max_idx)
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(self.emb_matrix), freeze=freeze)
            print(f'Embeddings loaded - shape is {self.emb_matrix.shape}')
            # sys.exit()
        else:
            self.embedding = nn.Embedding(num_features, embedding_dim)

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
    
class ReviewMLP_Embed_Classifier(Embedding, nn.Module):
    def __init__(self, num_features, num_classes, hidden_layer_dim=None, activation_fn = 'RELU', 
                    embedding_dim=100, embedding_type='pre-trained', embedding_file_name=None, 
                    word_to_index=None, max_idx=1000, freeze=True, **kwargs):
        
        super(ReviewMLP_Embed_Classifier, self).__init__(num_features=num_features, embedding_dim=embedding_dim, 
                    embedding_type=embedding_type, embedding_file_name=embedding_file_name, 
                    word_to_index=word_to_index, max_idx=max_idx, freeze=freeze, **kwargs)
        
        
        input_features = embedding_dim
        layers = []

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


class ReviewCNN_Embed_Classifier(Embedding, nn.Module):
    def __init__(self, num_features, num_classes, channel_list, activation_fn = 'RELU', max_pool=False,
                    embedding_dim=100, embedding_type=None, embedding_file_name=None, 
                    word_to_index=None, max_idx=1000, freeze=True, batch_norm=False, dropout=False, **kwargs):

        super(ReviewCNN_Embed_Classifier, self).__init__(num_features=num_features, embedding_dim=embedding_dim, 
                    embedding_type=embedding_type, embedding_file_name=embedding_file_name, 
                    word_to_index=word_to_index, max_idx=max_idx, freeze=freeze, **kwargs)
        
        layers = []
        
        in_channels = embedding_dim

        if channel_list is not None:
            for out_channels in channel_list:
                layers.append(nn.Conv1d(in_channels=in_channels,
                                        out_channels=out_channels, kernel_size=3))
                if activation_fn == 'RELU':
                    layers.append(nn.ReLU())
                if activation_fn == 'ELU':
                    layers.append(nn.ELU())
                if max_pool:
                    layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(out_channels))
                if dropout:
                    layers.append(nn.Dropout(p=0.5))
                
                in_channels = out_channels
              
        self.cnn_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels, num_classes) 

    def forward(self, x_in, apply_sigmoid=False):
        # print(f'x_input shape is {x_in.size()}')
        x_in = x_in.to(torch.long)
        embed_out = self.embedding(x_in) # => batch_size x seq_len x emb_dim
        embed_out = embed_out.permute(0, 2, 1) # => batch_size x emb_dim x seq_len
        # print(f'embedding shape is {embed_out.size()}')
        # embed_out,_ = torch.max(embed_out, dim=2)
        # print(f'embedding shape is {embed_out.size()}')
        
        # sys.exit()

        cnn_output = self.cnn_layers(embed_out.float())
        # print(f'cnn output shape is {cnn_output.size()}')
        cnn_output = torch.mean(cnn_output, dim=2)
        # print(f'cnn output shape is {cnn_output.size()}')
        # sys.exit()
        logits = self.fc(cnn_output).squeeze()

        if apply_sigmoid:
            logits = torch.sigmoid(logits)
        return logits


class ReviewRNN_Embed_Classifier(Embedding, nn.Module):
    def __init__(self, num_features, num_classes, rnn_hidden_size, activation_fn = 'RELU', 
                    embedding_dim=100, embedding_type=None, embedding_file_name=None, 
                    word_to_index=None, max_idx=1000, freeze=True, batch_norm=False, batch_first=True, dropout=False, **kwargs):

        super(ReviewRNN_Embed_Classifier, self).__init__(num_features=num_features, embedding_dim=embedding_dim, 
                    embedding_type=embedding_type, embedding_file_name=embedding_file_name, 
                    word_to_index=word_to_index, max_idx=max_idx, freeze=freeze, **kwargs)
        
        self.rnn = nn.GRU(input_size=embedding_dim,
                             hidden_size=rnn_hidden_size,
                             batch_first=batch_first)
        self.fc1 = nn.Linear(in_features=rnn_hidden_size,
                         out_features=rnn_hidden_size)
        self.fc2 = nn.Linear(in_features=rnn_hidden_size,
                          out_features=num_classes)

    def forward(self, x_in, x_lengths=None, apply_sigmoid=False):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, input_dim)
            x_lengths (torch.Tensor): the lengths of each sequence in the batch.
                They are used to find the final vector of each sequence
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        # print(f'x_input shape is {x_in.size()}')
        x_in = x_in.to(torch.long)
        x_embed = self.embedding(x_in) # => batch_size x seq_len x emb_dim
        
        # print(f'embedding shape is {embed_out.size()}')
        # embed_out,_ = torch.max(embed_out, dim=2)
        # print(f'embedding shape is {embed_out.size()}')
        
        # sys.exit()

        y_out, _ = self.rnn(x_embed.float())

        # print(f'cnn output shape is {cnn_output.size()}')
        # print(f'x_lengths shape is {x_lengths.size()}')
        if x_lengths is not None:
            y_out = column_gather(y_out, x_lengths)
        else:
            y_out = y_out[:, -1, :]

        # print(f'After column gather: y_out shape is {y_out.size()}')
        y_out = F.relu(self.fc1(F.dropout(y_out, 0.5)))
        logits = self.fc2(F.dropout(y_out, 0.5)).squeeze()

        if apply_sigmoid:
            logits = torch.sigmoid(logits)
        return logits



if __name__ == '__main__':
    rnn_hidden_size = 200
    RnnClassifier = ReviewRNN_Embed_Classifier(200, 1, rnn_hidden_size)
    print(RnnClassifier)
