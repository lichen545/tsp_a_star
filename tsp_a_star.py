import sys
import argparse
import timeit
import math
import heapq


# data structure for reading in file and converting to a graph
class Graph:
    '''
    {
        (1,2): [(2,3),(3.2)],
        (3,3): [(2,3)],
        ...
    }
     '''
     # adjacency list for all edges
    def __init__(self):
        self.edges = {}

    def neighbors(self, point):
        return self.edges[point]

    def cost(self, from_node, to_node):
        return mahattan_distance(from_node, to_node)

# wrapper for heapq
class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

def mahattan_distance(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

# used to calculate cost of distance from here to there
# def heuristic(a, b):
#     pass

def rectilinear(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def euclidean(a, b):
    (x_1, y_1) = a
    (x_2, y_2) = b
    return math.sqrt(((x_1 - x_2) ** 2) + ((y_1 - y_2) ** 2))

# implemented using a priority queue (pop node from frontier, add neighbors of node to frontier, repeat)
def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        # add all unexplored neighbors of current node to priority queue
        for nxt in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, nxt)
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                # nodes with lower from_cost + heuristic_cost have higher priority (lower number)
                priority = new_cost + heuristic(goal, nxt)
                frontier.put(nxt, priority)
                came_from[nxt] = current

    return came_from, cost_so_far

def held_karp(inputcities):
    # Choose an arbitrary start point
    startpt = inputcities[0]
    #
    cities = set(inputcities[1:])

    lookuptable = {}

    def min_dist(loc, path):
        """
        Minimum distance starting at city startpt, visiting all cities in path,
        and finishing at city loc
        """
        if (loc, path) not in lookuptable:

            if len(path) == 1 and loc in path:
                lookuptable[(loc, path)] = euclidean(startpt, loc)
            else:
                # Enumerate all routes to this point
                newpath = path - loc
                # Find minimum
                lookuptable[(loc, path)] = min(
                    [min_dist(newpath, x) + euclidean(x, loc)
                     for x in newpath])

        return lookuptable[(loc, path)]

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--heuristic', nargs='+', required=True, type=str)
    parser.add_argument('--problem', required=True, type=str)
    args = parser.parse_args()

    with open(args.problem, 'rb') as problemfile:
        lines = problemfile.readlines()

    input = Graph()
    cities = set()
    for line in lines[1:]:
        x, y = line.split()
        cities.add((x,y))


if __name__ == "__main__":
    main(sys.argv[1:])
