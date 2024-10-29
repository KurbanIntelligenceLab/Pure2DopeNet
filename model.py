import torch
import torch.nn as nn


class Pure2DopeNet(nn.Module):
    def __init__(self, input_channels=3, text_embedding_dim=768, feature_dim=16):
        super(Pure2DopeNet, self).__init__()

        # Convolutional Layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Conv2d(64, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

        # Text Embedding Layers
        self.text_embedding = nn.Sequential(
            nn.Linear(text_embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True)
        )

        # Fully Connected Layers with Concatenated Text Embeddings
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim * 8 * 8 + feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(512 + feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(256 + feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(128 + feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(64 + feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(32 + feature_dim, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(16 + feature_dim, 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(8 + feature_dim, 1)  # Final Output Layer
        )

    def forward(self, x, text_vector):
        # Embed text vector
        embedded_text = self.text_embedding(text_vector)

        # Pass through Convolutional Layers
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layers

        # Concatenate text embedding at each stage
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear) and layer.in_features != 1:
                x = torch.cat((x, embedded_text), dim=1)
            x = layer(x)

        return x
