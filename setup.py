from setuptools import setup, find_packages

setup(
    name="GPTNeoXColab",
    version="0.1",
    package_dir={"": "src"},  # Specify src as the root directory for packages
    packages=find_packages(where="src"),  # Search for packages in src
)