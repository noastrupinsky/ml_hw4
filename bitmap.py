import kagglehub
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np

# Download latest version
path = kagglehub.dataset_download("karnikakapoor/digits")
print("Path to dataset files:", path)

#Path to dataset files: /Users/chanabialik/.cache/kagglehub/datasets/karnikakapoor/digits/versions/4

root = "/content/digits_data/digits updated/digits updated"

bitmap_data = {}  # digit -> list of 7x12 bitmaps

for digit in sorted(os.listdir(root)):
    folder = os.path.join(root, digit)
    if not os.path.isdir(folder):
        continue

    bitmap_data[digit] = []

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        # --- 1. Load image ---
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # --- 2. Normalize values to [0,1] ---
        img = img / 255.0

        # --- 3. Resize to 7 × 12 (height=7, width=12) ---
        resized = cv2.resize(img, (12, 7), interpolation=cv2.INTER_AREA)

        # --- 4. Convert to bitmap (binary) ---
        # Use Otsu thresholding
        _, bitmap = cv2.threshold((resized * 255).astype(np.uint8),
                                  0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        bitmap_data[digit].append(bitmap)

centers = np.zeros((10, 84))

for d in range(10):
    digit_str = str(d)
    bitmaps = bitmap_data[digit_str]  # list of 7x12 arrays

    # flatten each bitmap to 84-dim vector
    flattened = [b.flatten() for b in bitmaps]

    # compute center = mean vector
    centers[d] = np.mean(flattened, axis=0)


class GaussianRBF(nn.Module):
    """
    Implements the Gaussian RBF module used in the original LeNet-5 paper.
    - Takes an 84-dimensional input (flattened 7x12 bitmap).
    - Computes RBF similarity to 10 fixed class centers.
    - Outputs a 10-dimensional vector of RBF activations.
    """

    def __init__(self, centers, sigma=3.0):
        """
        centers: numpy array or torch tensor of shape (10, 84)
                 Each row is the center for one digit class.
        sigma:   Gaussian bandwidth parameter.
        """
        super().__init__()

        # Convert centers to tensor of shape (10,84)
        centers = torch.tensor(centers, dtype=torch.float32)

        # Register them as *buffers* (not learnable parameters)
        self.register_buffer("centers", centers)

        # Store σ² and factor for the RBF exponent
        self.sigma = sigma
        self.sigma_sq = sigma ** 2

    def forward(self, x):
        """
        x: tensor shape (batch_size, 84)
        Returns: tensor shape (batch_size, 10)
        """

        # Expand x to shape (batch_size, 1, 84)
        # Expand centers to shape (1, 10, 84)
        # Then squared difference broadcasts to (batch_size, 10, 84)
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)

        # Compute squared Euclidean distance
        dist_sq = torch.sum(diff ** 2, dim=2)  # -> (batch_size, 10)

        # Gaussian RBF activation
        rbf_out = torch.exp(-dist_sq / (2 * self.sigma_sq))

        return rbf_out
