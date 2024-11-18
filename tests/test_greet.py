# Import the function from the installed module
from GPTNeoXColab import greet
from GPTNeoXColab import utils


# Use the function
print(greet("Alice"))


# Access a function from utils.py
result = utils.is_colab()
print("Is Colab:", result)