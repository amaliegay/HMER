from setuptools import Command, find_packages, setup

install_requires = ["numpy==1.24.3", "scikit-learn==1.2.2"]

setup(
    name="hmer",
    version="0.0.2",
    author="Amalie",
    author_email="amaliegay@outlook.com",
    description="Handwritten Mathematical Expression Recognition with Deep Learning",
    keywords="deep learning pytorch",
    url="https://github.com/amaliegay/HMER",
    install_requires=install_requires,
)
