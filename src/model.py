import torch
import torch.nn as nn

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

if __name__ == '__main__':
    hidden_layer_dim = [100, 50]
    mlpClassifier = ReviewMLPClassifier(200, 1, hidden_layer_dim)
    print(mlpClassifier)
