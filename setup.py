from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="GPTNeoXColab",
    version="0.1",
    package_dir={"": "src"},  # Specify src as the root directory for packages
    packages=find_packages(where="src"),  # Search for packages in src
    install_requires=required,
)
