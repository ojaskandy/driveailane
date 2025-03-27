from setuptools import setup, find_packages

setup(
    name="drivesafe",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "python-dotenv>=1.0.0",
        "pytest>=7.0.0",
    ],
    python_requires=">=3.8",
) 