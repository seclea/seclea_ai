try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import find_packages, setup

from __version__ import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name="seclea_ai",
    version=__version__,
    author="octavio",
    author_email="octavio.delser@gmail.com",
    description="Seclea integration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seclea/seclea_ai",
    packages=find_packages(exclude=["**test", "**example_files"]),
    package_data={
        "seclea_ai": [
            "lib/seclea_utils/clib/pigz.exe",
            "lib/seclea_utils/clib/pigz/zopfli/src/zopfli/*",
            "lib/seclea_utils/clib/pigz/zopfli/*",
            "lib/seclea_utils/clib/pigz/*",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "dill>=0.3.4",
        "pandas>=1.3.0",
        "pickleDB>=0.9.2",
        "requests>=2.0.0",
        "zstandard>=0.15.2",
    ],
)
