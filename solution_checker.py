#!/usr/bin/env python

import numpy as np


def readInstance(fp):
	""" Read rail* instance from ORLIB 
	
	Format is:
	number of rows (m), number of columns (n)
	for each column j (j=1,...,n): the cost of the column, the
	number of rows that it covers followed by a list of the rows 
	that it covers
	"""
	m, n = [int(x) for x in fp.readline().split()]
	assert(m >= 1)
	assert(n >= 1)
	obj = np.zeros(n, dtype=np.int32)
	matrix = np.zeros((m,n), dtype=np.int32)
	for j in range(n):
		numbers = [int(x) for x in fp.readline().split()]
		assert(len(numbers) >= 2)
		cj = numbers[0]
		count = numbers[1]
		col = numbers[2:]
		assert(count == len(col))
		obj[j] = cj
		for elem in col:
			assert(1 <= elem and elem <= m)
			matrix[elem-1][j] = 1
	return (obj, matrix)


def readSolution(fp):
	""" Read set covering solution

	Format is:
	objvalue
	j_1 j_2 ... j_k

	where j_1,...,j_k are the 0-based indices
	of the columns in the solution
	"""
	primalBound = int(fp.readline())
	assert(primalBound > 0)
	sol = [int(x) for x in fp.readline().split()]
	return (primalBound, sol)


def checker(objective, matrix, primalBound, solution):
	"""Check that the solution is feasible and the objective value matches"""
	# scatter solution to dense array
	m,n = matrix.shape
	denseSol = np.zeros(n, dtype=np.int32)
	for j in solution:
		denseSol[j] = 1
	# check objective value
	tot = np.dot(denseSol, objective)
	print("Objective value: {}".format(tot == primalBound))
	# check feasibility
	covered = np.matmul(matrix, denseSol)
	print("Feasibility: {}".format(all(covered)))


if __name__ == '__main__':
	import sys
	obj, matrix = readInstance(open(sys.argv[1]))
	m,n = matrix.shape
	print("Matrix of size {} x {}".format(m ,n))
	print("#nz:", np.count_nonzero(matrix))
	primalBound, sol = readSolution(open(sys.argv[2]))
	print("primalBound:", primalBound)
	checker(obj, matrix, primalBound, sol)
