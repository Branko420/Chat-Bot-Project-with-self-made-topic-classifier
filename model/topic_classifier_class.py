import torch.nn as nn

class TopicClassifier(nn.Module):
    def __init__(self, features_size, num_classes):
        super(TopicClassifier, self).__init__()
        self.fc = nn.Linear(features_size, num_classes)
    def forward(self, x):
        return self.fc(x)