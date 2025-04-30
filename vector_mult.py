import pandas as pd

vector1 = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
vector2 = pd.DataFrame({'A': [10, 20, 30, 40, 50]})

product = vector1 * vector2

print("Vector 1:")
print(vector1)
print("\nVector 2:")
print(vector2)
print("\nElement-wise Product:")
print(product)