import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, feature_dim=64, n_classes=2):
        super().__init__()

        self.net = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, input_dim, feature_dim=64, n_classes=2):
        super().__init__()

        # θ → dibagikan ke server
        self.theta = FeatureExtractor(
            input_dim=input_dim,
            feature_dim=feature_dim
        )

        # φ → tetap lokal
        self.phi = Classifier(
            feature_dim=feature_dim,
            n_classes=n_classes
        )

    def forward(self, x):
        feature = self.theta(x)
        output = self.phi(feature)

        return output

    def extract_features(self, x):
        return self.theta(x)


def build_model(input_dim, feature_dim=64, n_classes=2):
    return Model(
        input_dim=input_dim,
        feature_dim=feature_dim,
        n_classes=n_classes
    )