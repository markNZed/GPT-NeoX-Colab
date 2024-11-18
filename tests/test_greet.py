# Import the function from the installed module
from gpt_neox_colab import greet
from gpt_neox_colab import utils


# Use the function
print(greet("Alice"))


# Access a function from utils.py
result = utils.is_colab()
print("Is Colab:", result)