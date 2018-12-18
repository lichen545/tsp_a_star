import itertools
import random
import sys


def held_karp(dists):
    """
    Takes in an distance matrix and returns a tuple, (cost, path)
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

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1
    return


if __name__ == '__main__':
    arg = sys.argv[1]

    if arg.endswith('.csv'):
        dists = read_distances(arg)
    else:
        dists = generate_distances(int(arg))

    # Pretty-print the distance matrix
    for row in dists:
        print(''.join([str(n).rjust(3, ' ') for n in row]))

    print('')

    print(held_karp(dists))