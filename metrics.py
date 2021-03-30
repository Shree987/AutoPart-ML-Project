import numpy as np
from math import log2, ceil
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


def CodeCost(D):
	return 0


def TotalEncodingCost(D, A):
	description_cost = DescriptionCost(A)
	code_cost = CodeCost(D)
	return description_cost + code_cost


if __name__ == "__main__":
	A = np.array([1, 2, 3, 4, 5])
	print(DescriptionCost(A))