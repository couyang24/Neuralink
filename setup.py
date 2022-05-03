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
        "Neuralink provides deeplearning modelling tools and is based on the methods from deeplearning.AI at Coursera."
    ),
    license="Apache License",
    keywords="Neuralink documentation tutorial",
    packages=["neuralink"],
    long_description=_read("README.md"),
    long_description_content_type="text/markdown",
)
