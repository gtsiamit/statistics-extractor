from setuptools import setup, find_packages

REQUIREMENTS_FILE = "requirements.txt"


def read_requirements():
    with open(REQUIREMENTS_FILE) as f:
        return f.read().splitlines()


setup(
    name="statistics_extractor",
    version="0.1.0",
    author="",
    description="",
    packages=find_packages(),  # finds all packages automatically
    install_requires=read_requirements(),  # uses dependencies from requirements.txt
    python_requires="~=3.12.0",
)
