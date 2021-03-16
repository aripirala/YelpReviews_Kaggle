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
