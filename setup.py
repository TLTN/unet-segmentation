from setuptools import setup, find_packages

setup(
    name="unet-segmentation",
    version="1.0.0",
    author="TLTN",
    author_email="tltnlovelala@gmail.com",
    description="U-Net implementation for image segmentation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "LICENSE :: OSI Approved :: MIT LICENSE",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
    install_requires=open("requirements.txt").read().splitlines(),
)