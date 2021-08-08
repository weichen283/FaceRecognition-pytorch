from torch import nn


class Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),    # 32, 32*32

            nn.Conv2d(32, 64, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),    # 64, 16*16

            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),    # 64, 8*8
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64*8*8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64*8*8)
        x = self.classifier(x)
        return x





