from setuptools import setup, find_packages

setup(
    name="cogvideox-factory",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "diffusers>=0.30.3",
        "transformers>=4.45.2",
        "pytest>=7.0.0",
    ],
    python_requires=">=3.8",
)
