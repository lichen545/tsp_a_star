import sys
import argparse
import timeit
import math
import heapq
import itertools
import pprint

# data structure for reading in file and converting to a graph
class Graph:
    ''' 
    # source and sink are equivalent to start_point (1 in this example)
    # (x, S): starting from 1, path min cost ends at vertex x, passing vertices in set S exactly once
    {   
        ("source", None): [ (2, ()), (3, ()), (4, ()) ] 
        (2, ()): [ (3, (2)), (4, (2)) ],
        ...,
        (3, (2)): [ (4, (2,3)) ],
        ...,
        (4, (2,3)): [ ("sink", (2,3,4)) ],
    }
    '''
    # adjacency list for graph, takes a start point and a list of cities as input
    def __init__(self, start_point, cities):
        self.start_point = start_point
        self.cities = set(cities)
        self.edges = {}
        # add source node with key "source"
        cities = set(cities)
        cities.discard(start_point)
        # create source node
        connected_values = [(c, ()) for c in cities]
        self.edges[('source', None)] = connected_values
        # add rest of the nodes to graph
        while connected_values:
            tmp = []
            for x,S in connected_values:
                # add x to S to get next S
                next_S = sorted(list(S) + [x])
                next_S = tuple(next_S)
                # find cities not in next_S and add them to adjacency list
                remainder_set = cities.copy() - set(next_S)
                next_values = [(next_x, next_S) for next_x in remainder_set]
                tmp += next_values
                # continue attaching nodes to previous nodes
                if tmp:
                    self.edges[(x,S)] = next_values
                # full set reached, connect to sink node
                else:
                    self.edges[(x,S)] = [("sink", next_S)]
            connected_values = tmp
    
    def __str__(self):
        return str(self.__dict__)

    # returns all neighbors of given node
    def neighbors(self, node):
        return self.edges[node]

    # return cost of edge (from_node, to_node)
    def cost(self, from_node, to_node):
        from_point = from_node[0]
        to_point = to_node[0]
        if from_point == "source":
            return rectilinear(self.start_point, to_point)
        elif to_point == "sink":
            return rectilinear(from_point, self.start_point)
        else:
            return rectilinear(from_point, to_point)

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

def rectilinear(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def euclidean(a, b):
    (x_1, y_1) = a
    (x_2, y_2) = b
    return math.sqrt(((x_1 - x_2) ** 2) + ((y_1 - y_2) ** 2))

# implementation of the held-karp algorithm for solving TSP
def held_karp(start_point, cities):
    cities = set(cities)
    cities.remove(start_point)
    cities = tuple(cities)

    # Quick and dirty memoization
    lookuptable = {}

    def min_dist(loc, path):
        """
        Minimum distance starting at city start_point, visiting all cities in path,
        and finishing at city loc
        """
        if (loc, path) not in lookuptable:

            if len(path) == 1 and loc in path:
                lookuptable[(loc, path)] = rectilinear(start_point, loc)
            else:
                # Enumerate all routes to this point
                newpath = path - loc
                # Find minimum
                lookuptable[(loc, path)] = min(
                    [min_dist(x, newpath) + rectilinear(x, loc)
                     for x in newpath])

        return lookuptable[(loc, path)]

    [min_dist(x, cities) for x in cities]

    return lookuptable

# heuristic for estimating cost of going from a node to goal state via taking the smallest distance and multiplying by number of cities left to visit
def naive_euclidean(graph, next):
    # find all nodes that have been traversed
    if next[0] != 'sink':
        traversed = next[1] + (next[0])
    # at goal state, cost is 0
    else:
        return 0
    # find all remaining points still to be explored
    remainder = graph.cities - set(traversed)
    pairs = list(itertools.combinations(remainder, 2))
    # print("===PAIRS===", pairs)
    distances = [euclidean(p1,p2) for p1,p2 in pairs]
    
    return min(distances) * len(remainder)

# heuristic for estimating cost of going from a node to goal state via taking the average of the remaining n smallest distances, where n = # of cities left to visit
def avg_remaining(graph, next):
    # find all nodes that have been traversed
    if next[0] != 'sink':
        traversed = next[1] + (next[0])
    # at goal state, cost is 0
    else:
        return 0
    # find all remaining points still to be explored
    remainder = graph.cities - set(traversed)
    pairs = list(itertools.combinations(remainder, 2))
    distances = [euclidean(p1,p2) for p1,p2 in pairs]
    distances.sort()

    return distances[:len(remainder)] / len(remainder)

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
        
        if current[0] == goal:
            break
        
        # add all unexplored neighbors of current node to priority queue
        for next in graph.neighbors(current):
            print("CURRENT:", current)
            print("NEXT:", next, end='\n\n')
            new_cost = cost_so_far[current] + graph.cost(current, next)
            # update cost of next neighbor if applicable
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                # nodes with lower from_cost + heuristic_cost have higher priority (lower number)
                priority = new_cost + avg_remaining(graph, next)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far


def main(args):
    # FUNCTION_MAP = {'naive_euclidean' : my_top20_func,
    #                 'avg_remaining' : my_listapps_func }

    parser = argparse.ArgumentParser()
    # parser.add_argument('--heuristic', nargs='+', required=True, type=str)
    parser.add_argument('--problem', required=True, type=str)
    args = parser.parse_args()

    with open(args.problem, 'rb') as file:
        lines = file.readlines()

    # exits if input file is empty
    if len(lines) <= 1:
        print("Error: empty input")
        exit(1)

    cities = []
    for line in lines[1:]:
        x, y = line.split()
        cities.append( (int(x), int(y)) )
    
    start = cities[0]
    # memo = held_karp(start, cities)
    a_star_graph = Graph(start, cities)
    print("HELD-KARP GRAPH:")
    pprint.pprint(a_star_graph.__dict__)
    print()

    came_from, cost_so_far = a_star_search(a_star_graph, ("source", None), "sink")
    print("FINAL PATH:")
    pprint.pprint(came_from)
    print()
    print("TOTAL COST:")
    pprint.pprint(cost_so_far)

if __name__ == "__main__":
    main(sys.argv[1:])