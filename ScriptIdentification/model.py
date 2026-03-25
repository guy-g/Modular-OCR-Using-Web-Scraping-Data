from Recognition.model import *


class ScriptClassifier(nn.Module):
    def __init__(self, scripts, in_channels=3):
        super().__init__()
        self.scripts = scripts
        self.feature_extractor = FeatureExtractor(in_channels)
        self.fc = nn.Linear(256, len(scripts))
        self.sigmoid_func = nn.Sigmoid()

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.mean(dim=1)
        out = self.fc(features)
        out = self.sigmoid_func(out)
        pred = {s: out[:, i] for i, s in enumerate(self.scripts)}
        return pred

