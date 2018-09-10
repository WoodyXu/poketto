from setuptools import setup, find_packages

setup(
    name="poketto",
    version="0.5.0",
    keywords=["machine learning", "metrics", "feature engineering"],
    long_description="Hopefully to be a collection of machine learning toolkits.",
    author="Xu Quan",
    author_email="thisiswoody.cs@gmail.com",
    packages=find_packages(exclude=["tests"])
)
