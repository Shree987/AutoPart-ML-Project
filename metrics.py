import numpy as np
from math import log2, ceil, log
import pandas as pd


def log_n(n):
	val = 0
	n = log2(n)
	while n>0:
		val = val + n
		n = log2(n)

	return val


def DescriptionCost(A):
	k = len(A)
	cost1 = log_n(k)
	cost2 = 0

	i = 0
	while i < k-1:
		j = i
		cost_sum = 0
		while j<k:
			cost_sum = cost_sum + A[j]
			j = j+1
		cost_sum += i+1-k
		cost2 = cost2 + ceil(log2(cost_sum))
		i = i + 1

	cost3 = 0

	for i in range(k):
		for j in range(k):
			cost3 = cost3 + ceil(log2((A[i]*A[j] + 1)))

	return cost1 + cost2 + cost3


def CodeCost(D, G, k):
	n = G.shape[0]
	cost = 0

	# Iterate for cluster ID
	for i in range(k):
		for j in range(k):
			total = 0
			ones = 0
			x = 0

			while x<n:
				while x<n and G[x] != i+1:
					x = x + 1
				if x == n:
					break

				y = 0
				while y<n:
					while y<n and G[y] != j+1:
						y = y + 1
					if y == n:
						break

					total = total + 1
					if D[x][y] == 1:
						ones = ones + 1

					y = y + 1
				x = x + 1

			P = ones / total

			if ones > 0 and ones < total:
				cost = cost - ones*log(P) - (total - ones)*log(1 - P)

	return cost


def TotalEncodingCost(D, A, G):
	description_cost = DescriptionCost(A)
	code_cost = CodeCost(D, G, len(A))
	return description_cost + code_cost


def OutlierDetection(D, A, G):
	n = G.shape[0]
	k = len(A)
	diff = 0.5
	outlier =[]
	D_ = D.copy()
	k_group = {}

	for i in range(n):
		for j in range(n):
			if i == j or D[i][j] == 0:
				continue
			(x, y) = (G[i], G[j])
			if x == y:
				continue
			k_group[(x, y)] = (i, j)


	for key in k_group.keys():
		(x, y) = k_group[key]
		D_[x][y] -= 1
		cost_ = TotalEncodingCost(D_, A, G) - TotalEncodingCost(D, A, G)
		D_[x][y] += 1
		# print(key, cost_)
		if abs(cost_) > diff:
			outlier.append(key)

	for KEY in outlier:
		(X , Y) = KEY
		RemoveOutliers(D, G, X, Y)

	return D


def RemoveOutliers(D, G, x, y):
	n = G.shape[0]
	for i in range(n):
		if G[i] == x:
			for j in range(n):
				if G[j] == y and D[i][j] == 1:	
					D[i][j] -= 1
					D[j][i] -= 1


if __name__ == "__main__":
	A = np.array([1, 2, 3, 4, 5])
	print(DescriptionCost(A))
