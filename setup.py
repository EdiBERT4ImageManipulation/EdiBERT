from setuptools import setup, find_packages

setup(
    name='edibert',
    version='0.0.1',
    description='EdiBERT, a generative model for image editing',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
