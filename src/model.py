# pylint: disable=no-member

import torch
import torch.nn as nn
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
    def __init__(self, num_features, num_classes, hidden_layer_dim=None, activation_fn = 'RELU', embedding_dim=100):
        super(ReviewMLP_Embed_Classifier, self).__init__()
        # self.fc1 = nn.Linear(num_features, num_classes)
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_features, embedding_dim)
        layers = []
        
        input_features = embedding_dim

        if hidden_layer_dim is not None:
            for hidden_features in hidden_layer_dim:
                layers.append(nn.Linear(input_features, hidden_features))
                if activation_fn == 'RELU':
                    layers.append(nn.ReLU())
                input_features = hidden_features
        layers.append(nn.Linear(input_features, num_classes))        
        self.seq_model = nn.Sequential(*layers)

    def forward(self, x_in, apply_sigmoid=False):
        # print(f'x_input shape is {x_in.size()}')
        x_in = x_in.to(torch.long)
        embed_out = self.embedding(x_in)
        embed_out = embed_out.permute(0, 2, 1)
        # print(f'embedding shape is {embed_out.size()}')
        embed_out = torch.mean(embed_out, dim=2)
        # print(f'embedding shape is {embed_out.size()}')
        
        # sys.exit()

        logits = self.seq_model(embed_out).squeeze()
        if apply_sigmoid:
            logits = torch.sigmoid(logits)
        return logits

if __name__ == '__main__':
    hidden_layer_dim = [50, 25]
    mlpClassifier = ReviewMLP_Embed_Classifier(200, 1, hidden_layer_dim)
    print(mlpClassifier)
