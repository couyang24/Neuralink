import os

from setuptools import setup


def _read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

exec(open("neuralink/_version.py").read())

setup(
    name="neuralink",
    version=__version__,
    author="Chengran (Owen) Ouyang",
    author_email="chengranouyang@gmail.com",
    description=(
        "This is a learning notes from deeplearning.AI at Coursera, which also"
        " provides basic DeepLearning tools."
    ),
    license="Apache License",
    keywords="Neuralink documentation tutorial",
    packages=["neuralink"],
    long_description=_read("README.md"),
)
