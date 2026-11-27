import cv2, numpy as np, torch, torch.nn as nn

CROP_TOP, CROP_BOTTOM = 110, 216   # rows kept from the 240-px frame (drops sky and likely hood)
OUT_W, OUT_H = 200, 66

def preprocess(bgr):
    """uint8 BGR frame (240x320x3) -> uint8 CHW array (3x66x200)."""
    crop = bgr[CROP_TOP:CROP_BOTTOM]
    small = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(small.transpose(2, 0, 1))

class PilotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ELU(),
            nn.Conv2d(24, 36, 5, 2), nn.ELU(),
            nn.Conv2d(36, 48, 5, 2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(1152, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):           # x: float in [0, 1], shape N x 3 x 66 x 200
        return self.head(self.features(x - 0.5)).squeeze(1)