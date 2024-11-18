from setuptools import setup, find_packages  # type: ignore[import]

# Read requirements from requirements.txt
# with open("requirements.txt") as f:
#    required = f.read().splitlines()

setup(
    name="gpt_neox_colab",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src")
    # install_requires=required,
)
