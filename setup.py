from setuptools import setup, find_packages

REQUIREMENTS_FILE = "requirements.txt"


def read_requirements():
    with open(REQUIREMENTS_FILE) as f:
        return f.read().splitlines()


setup(
    name="statistics_extractor",
    version="0.1.0",
    author="",
    description="A statistics extractor for tabular data.",
    packages=find_packages(),  # finds all packages automatically
    install_requires=read_requirements(),  # uses dependencies from requirements.txt
    extras_require={
        "dev": ["pytest==8.3.4"],  # δependencies for development/testing
    },
    python_requires="~=3.12.0",
)
