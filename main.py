import argparse
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import FloatTensor, LongTensor
import torchvision

from dataset import dataset
from decoder import Decoder
from encoder import Encoder

# Load number of samples
n_samples = 10

# Manual seed
random_state = 258

# Table of image name, image, and the expression in the image
file_names, features, labels = dataset(n_samples, random_state=random_state)

# Turn image data into tensors
for feature in features:
    feature = torchvision.transforms.ToTensor()(feature)

# Split data into training and test sets
# 20% of data will be test and 80% will be train
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=random_state
)

# Building a model

# Device-agnostic code
parser = argparse.ArgumentParser(description="Device-agnostic code")
parser.add_argument("--disable-cuda", action="store_true", help="Dsiable CUDA")
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device("cuda")
    print("")
else:
    args.device = torch.device("cpu")


# Construct a model that subclasses nn.Module
class BTTR(nn.Module):
    def __init__(
        self,
        # Encoder
        growth_rate: int,
    ):
        super().__init__()
        # Create nn layers
        self.encoder = Encoder(growth_rate=growth_rate)
        self.decoder = Decoder()

    def forward(self, img: FloatTensor, tgt: LongTensor) -> FloatTensor:
        feature = self.encoder(img)
        out = self.decoder(feature, tgt)
