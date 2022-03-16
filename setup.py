import os
from setuptools import setup

def _read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="deeplearning",
    version="0.0.1",
    author="Chengran (Owen) Ouyang",
    author_email="chengranouyang@gmail.com",
    description=(
        "This is a learning notes from deeplearning.AI at Coursera, which also"
        " provides basic DeepLearning tools."
    ),
    license="MIT",
    keywords="DeepLearning documentation tutorial",
    packages=["deeplearning"],
    long_description=_read("README.md"),
)
