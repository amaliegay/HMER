# Handwritten Mathematical Expression Recognition

## Description   
Convert offline handwritten mathematical expression to LaTeX sequence

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/amaliegay/HMER

# install project   
cd HMER
conda create --yes --name hmer python=3.11
conda activate hmer
conda install --yes pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda==11.8 --channel pytorch --channel nvidia
pip install -e .
```