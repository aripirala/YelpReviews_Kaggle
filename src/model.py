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

class ReviewMLPEmbClassifier(nn.Module):
    def __init__(self, embedding_sizes, num_features, num_classes, hidden_layer_dim=None, activation_fn = 'RELU'):
        super(ReviewMLPEmbClassifier, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        self.n_emb = sum(e.embedding_dim for e in self.embeddings)
        # self.lin1 = nn.Linear(self.n_emb, 100)
        # self.fc1 = nn.Linear(num_features, num_classes)
        layers = []
        input_features = self.n_emb
        if hidden_layer_dim is not None:
            for hidden_features in hidden_layer_dim:
                layers.append(nn.Linear(input_features, hidden_features))
                if activation_fn == 'RELU':
                    layers.append(nn.ReLU())
                input_features = hidden_features
        layers.append(nn.Linear(input_features, num_classes))        
        self.model = nn.Sequential( *layers)

    def forward(self, x_in, apply_sigmoid=False):
        x_in = [e(x_in[i]) for i,e in enumerate(self.embeddings)]
        x_in = torch.cat(x_in, 0)
        logits = self.model(x_in).squeeze()
        if apply_sigmoid:
            logits = torch.sigmoid(logits)
        return logits

if __name__ == '__main__':
    hidden_layer_dim = [100, 50]
    embedded_cols = {f'col_{i}': 7497 for i in range(10)}
    embedding_sizes = [(n_categories, min(500, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]
    # embedding_sizes
    embedded_col_names = embedded_cols.keys()
    
    mlpClassifier = ReviewMLPEmbClassifier(embedding_sizes, 200, 1, hidden_layer_dim)
    print(mlpClassifier)
    
    one_hot_sample = [6, 356, 0, 6, 222, 9, 357, 49, 44, 6]
    x = torch.tensor(one_hot_sample)
    print(mlpClassifier(x))
