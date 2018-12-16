import itertools
import sys
from random import randint

MIN = 0
MAX = 500

def generate_input(n, filename):
    with open(filename, "w") as f:
        f.write("{n}\n".format(n=n))
        for _ in itertools.repeat(None, n):
            f.write("{p0} {p1}\n".format(p0=randint(MIN, MAX), p1=randint(MIN, MAX)))

if __name__ == "__main__":
    n = int(sys.argv[1])
    filename = "tsp{}.txt".format(n)
    generate_input(n, filename)
