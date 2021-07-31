import pandas as pd
from src.ann import *


filename = input("Data filename: ")
data = pd.read_csv(filename)

filename = input("Test filename: ")
test = pd.read_csv(filename)

learning_rate = input("Learning rate: ")
episodes = input("episodes: ")
ann(data, "target", [10, 10], 0.01, 100, True, test)

