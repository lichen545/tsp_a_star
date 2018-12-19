import itertools
import random
import sys
import pprint
import time

from utils import *

def held_karp(dists, cities):
    """
    Takes in an adjacency matrix and returns a tuple, (cost, path)
    """
    n = len(dists)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    # grab all bits except the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for _ in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)
    # correct ordering and convert to coordinates
    path.reverse()
    return opt, list(map(lambda x: cities[x], path))


def main(arg):
    with open(arg, 'rb') as input_file:
        lines = input_file.readlines()

    # exits if input file is empty
    if len(lines) <= 1:
        print("Error: empty input")
        exit(1)
    
    # read input and store list of cities
    cities = set()
    for line in lines[1:]:
        x, y = line.split()
        cities.add( (int(x), int(y)) )
    cities = list(cities)
    
    # start tracking execution time
    start_time = time.time()

    # convert to adjacency matrix
    adjacency_matrix = [[rectilinear(row_city, col_city) for col_city in cities] for row_city in cities]

    # Pretty-print the distance matrix
    print("ADJACENCY MATRIX:")
    for row in adjacency_matrix:
        print(row)
    print()

    cost, path = held_karp(adjacency_matrix, cities)
    print("FINAL PATH:", path)
    print("TOTAL COST:", cost)
    print()

    # print total execution time
    time_elapsed = time.time() - start_time 
    print_standardized_time(time_elapsed)

if __name__ == "__main__":
    main(sys.argv[1])