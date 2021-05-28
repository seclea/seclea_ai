try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="seclea_ai",
    version="0.0.12",
    author="octavio",
    author_email="octavio.delser@gmail.com",
    description="Seclea integration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seclea/seclea_ai",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[],
)
