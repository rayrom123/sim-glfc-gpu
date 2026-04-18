import torch.nn as nn

class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        if hasattr(feature_extractor, 'feature_dim'):
            self.fc = nn.Linear(feature_extractor.feature_dim, numclass, bias=True)
        else:
            self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)

    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self, inputs):
        return self.feature(inputs)

    def predict(self, fea_input):
        return self.fc(fea_input)


class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

class MLP_FeatureExtractor(nn.Module):
    def __init__(self, in_dim=33, hidden=128):
        super(MLP_FeatureExtractor, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
        )
        self.feature_dim = 64

    def forward(self, x):
        return self.body(x)

class MLP_Encoder(nn.Module):
    def __init__(self, in_dim=33, hidden=128, num_classes=100):
        super(MLP_Encoder, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.body(x)
        out = self.fc(out)
        return out

class CNN_FeatureExtractor(nn.Module):
    def __init__(self, in_dim=33):
        super(CNN_FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.feature_dim = 64

    def forward(self, x):
        # x shape: (batch, 33) -> reshape thành (batch, 1, 33)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        return out

class CNN_Encoder(nn.Module):
    def __init__(self, in_dim=33, num_classes=100):
        super(CNN_Encoder, self).__init__()
        self.feature = CNN_FeatureExtractor(in_dim)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.feature(x)
        out = self.fc(out)
        return out
