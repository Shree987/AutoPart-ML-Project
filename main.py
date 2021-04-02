import numpy as np
import pandas as pd
from metrics import TotalEncodingCost, OutlierDetection


# Import the Adjacency Matrix

inputExcel = pd.read_excel("AdjacencyMatrix.xlsx", header=None)
D = inputExcel.to_numpy()

print("Adjacency matrix shape: ", D.shape)

A = np.array([2, 3, 2])
G = np.array([1, 1, 2, 2, 2, 3, 3])

print("Total encoding cost = ",TotalEncodingCost(D, A, G))

# Process to map node to k-cluster





# Oulier Detection

D_ = D.copy()
D_ = OutlierDetection(D_, A, G)
outlier = []
n = G.shape[0]
for i in range(n):
	for j in range(n):
		if D_[i][j] == 0 and D[i][j] == 1:
			outlier.append((i, j))
		j = j + 1

if len(outlier) == 0:
	print("\nNo outlier detected!")

else:
	print("\nFollowing are outliers detected\n")
	no = 1
	for edge in outlier:
		(x, y) = edge
		print("Edge %-6d : %6d    <-> %6d" % (no, x+1, y+1))
		no = no + 1

df = pd.DataFrame (D_)
filepath = 'FinalAdjacencyMatrix.xlsx'
df.to_excel(filepath, index=False, header = False)