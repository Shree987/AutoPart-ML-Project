import numpy as np
import pandas as pd
from metrics import TotalEncodingCost


# Import the Adjacency Matrix
inputExcel = pd.read_excel("AdjacencyMatrix.xlsx", header=None)
D = inputExcel.to_numpy()

print("Adjacency matrix shape: ", D.shape)

# Process to map node to k-cluster

mapping = {0:1, 1:3, 2:3, 3:2, 4:2, 5:2, 6:1}
A = np.array([2, 3, 2])
G = np.array([1, 3, 3, 2, 2, 2, 1])
#print(mapping, type(mapping))

print("Total encoding cost = ",TotalEncodingCost(D, A, G))












