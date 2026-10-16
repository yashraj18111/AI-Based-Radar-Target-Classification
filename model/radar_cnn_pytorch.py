# radar_cnn_pytorch.py
# CNN model for radar target classification
# built this following the pytorch docs and some youtube tutorials, then adapted for spectrograms
#
# there are two classifiers here:
# 1. RadarTargetCNN  - the real pytorch CNN (if pytorch loads correctly)
# 2. RadarNeuralClassifier - sklearn MLP fallback (for when pytorch DLLs fail on windows)

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier

import config

logger = logging.getLogger(__name__)

# try loading pytorch - on some windows machines the C++ DLLs fail to load even if
# the import itself succeeds. ran into this on college lab PCs. the dummy tensor
# test catches that case before anything breaks.
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _ = torch.zeros(1)  # if this fails, pytorch isnt actually working
    TORCH_AVAILABLE = True
    logger.info("pytorch loaded fine")
except Exception as err:
    logger.warning(f"pytorch failed to load ({err}), using sklearn fallback")


if TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        """One convolutional block: Conv -> BN -> Conv -> BN -> Pool -> Dropout"""

        def __init__(self, in_ch, out_ch, dropout_rate=0.2):
            super().__init__()
            # two 3x3 convs with padding=1 so spatial size stays the same before pooling
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(out_ch)
            self.pool  = nn.MaxPool2d(2, 2)  # halves spatial size: 128->64->32->16
            # Dropout2d drops whole feature maps - better than per-pixel dropout for conv layers
            self.dropout = nn.Dropout2d(p=dropout_rate)
            # LeakyReLU instead of ReLU - tried ReLU but some neurons were dying
            # (always outputting 0). LeakyReLU with slope=0.1 lets a small gradient through
            # even for negative inputs so neurons can recover
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        def forward(self, x):
            x = self.act(self.bn1(self.conv1(x)))
            x = self.act(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = self.dropout(x)
            return x


    class RadarTargetCNN(nn.Module):
        """
        CNN for classifying radar spectrograms as car / drone / person.
        Architecture: 3 conv blocks -> extra conv -> global avg pool -> 2 FC layers
        """

        def __init__(self, in_channels=config.INPUT_CHANNELS, num_classes=config.NUM_CLASSES):
            super().__init__()

            # 3 blocks doubling channels each time: 32 -> 64 -> 128
            # this is the standard VGG-style pattern - more channels = more features learned
            # dropout increases with depth since deeper layers overfit more easily
            self.block1 = ConvBlock(in_channels, 32,  dropout_rate=0.15)
            self.block2 = ConvBlock(32, 64,           dropout_rate=0.20)
            self.block3 = ConvBlock(64, 128,          dropout_rate=0.25)

            # one more conv to get to 256 channels, no pooling here
            self.conv4  = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
            self.bn4    = nn.BatchNorm2d(256)
            self.act4   = nn.LeakyReLU(negative_slope=0.1, inplace=True)

            # AdaptiveAvgPool collapses whatever spatial size we have down to 1x1
            # cleaner than hardcoding the flatten size which breaks if input size changes
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

            # classifier: 256 -> 128 -> 3 classes
            self.fc1        = nn.Linear(256, 128)
            self.fc_bn      = nn.BatchNorm1d(128)
            self.fc_dropout = nn.Dropout(p=0.4)  # FC layers overfit easily, so higher dropout
            self.fc2        = nn.Linear(128, num_classes)  # raw logits, no softmax needed

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.act4(self.bn4(self.conv4(x)))
            x = self.global_pool(x)
            x = torch.flatten(x, 1)  # (batch, 256, 1, 1) -> (batch, 256)
            x = F.relu(self.fc_bn(self.fc1(x)))
            x = self.fc_dropout(x)
            return self.fc2(x)


class RadarNeuralClassifier:
    """
    Sklearn MLP classifier wrapper - used as fallback when pytorch isnt available,
    and also what actually gets saved/loaded since sklearn models serialize easily with pickle.

    Hidden layers (256, 128, 64) mirror the FC head of the CNN so the architecture
    is at least conceptually consistent between the two.
    """

    def __init__(self, hidden_layer_sizes=(256, 128, 64), max_iter=200,
                 random_state=config.SEED):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state

        # sklearn MLP - solver=adam works way better than sgd on spectrogram features
        # alpha=1e-4 is L2 regularization to prevent overfitting
        # adaptive lr means it automatically reduces lr when training slows down
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=config.BATCH_SIZE,
            learning_rate="adaptive",
            learning_rate_init=config.LEARNING_RATE,
            max_iter=max_iter,
            early_stopping=True,       # stops if val accuracy doesnt improve
            validation_fraction=0.15,  # holds out 15% internally for early stopping
            n_iter_no_change=5,        # stop after 5 rounds of no improvement
            random_state=random_state,
            verbose=True,
        )

    def fit(self, X, y):
        """Train on spectrogram images. X is (N, H, W, C) uint8, y is class indices."""
        # flatten images to 1D and normalize to [0,1] - sklearn cant handle 3D input
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        self.model.fit(X_flat, y)
        return self

    def predict_proba(self, X):
        """Returns probability distribution over classes for each input."""
        if X.ndim == 3:
            X = np.expand_dims(X, axis=0)  # single image -> add batch dimension
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        return self.model.predict_proba(X_flat)

    def predict(self, X):
        """Returns predicted class index for each input."""
        if X.ndim == 3:
            X = np.expand_dims(X, axis=0)
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        return self.model.predict(X_flat)


def save_neural_classifier(classifier, filepath=None):
    """Save trained model to disk using pickle."""
    if filepath is None:
        filepath = config.MODEL_OUTPUT_DIR / "best_radar_model.pkl"
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(classifier, f)
    logger.info(f"model saved to {path}")


def load_neural_classifier(filepath=None):
    """Load a saved model from disk. Returns None if file doesnt exist."""
    if filepath is None:
        filepath = config.MODEL_OUTPUT_DIR / "best_radar_model.pkl"
    path = Path(filepath)
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        classifier = pickle.load(f)
    logger.info(f"model loaded from {path}")
    return classifier
