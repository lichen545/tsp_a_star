import sys
import argparse
import math
import heapq
import itertools
import pprint
import time
import operator
from collections import OrderedDict

from utils import *

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
        self.edges[(start_point, None)] = connected_values
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
                    self.edges[(x,S)] = [(start_point, next_S)]
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

# heuristic for estimating cost of going from a node to goal state via taking the smallest distance and multiplying by r = number of cities left to visit 
def naive_euclidean(graph, next):
    # find all nodes that have been traversed
    if next[0] != graph.start_point:
        traversed = next[1] + (next[0])
    # at goal state, cost is 0
    else:
        return 0
    # find all remaining points still to be explored
    remainder = graph.cities - set(traversed)
    pairs = list(itertools.combinations(remainder, 2))
    # print("===PAIRS===", pairs)
    distances = [euclidean(p1,p2) for p1,p2 in pairs]
    
    r = len(remainder)
    return min(distances) * r

# heuristic for estimating cost of going from a node to goal state via taking the average of the remaining r smallest distances, where r = number of cities left to visit
def avg_remaining(graph, next):
    # find all nodes that have been traversed
    if next[0] != graph.start_point:
        traversed = next[1] + (next[0])
    # at goal state, cost is 0
    else:
        return 0
    # find all remaining points still to be explored
    remainder = graph.cities - set(traversed)
    pairs = list(itertools.combinations(remainder, 2))
    distances = [rectilinear(p1,p2) for p1,p2 in pairs]
    distances.sort()

    r = len(remainder)
    return sum(distances[:r]) / r

# implemented using a priority queue (pop node from frontier, add neighbors of node to frontier, repeat)
def a_star_search(graph, start, goal, heuristic):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = OrderedDict()
    cost_so_far = OrderedDict()
    came_from[start] = None
    cost_so_far[start] = 0
    
    iter = 0
    while not frontier.empty():
        current = frontier.get()
        
        if current[0] == goal and iter != 0:
            break
        
        # add all unexplored neighbors of current node to priority queue
        for next in graph.neighbors(current):
            # print("CURRENT:", current)
            # print("NEXT:", next, end='\n\n')
            new_cost = cost_so_far[current] + graph.cost(current, next)
            # update cost of next neighbor if applicable
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                # nodes with lower from_cost + heuristic_cost have higher priority (lower number)
                priority = new_cost + heuristic(graph, next)
                frontier.put(next, priority)
                came_from[next] = current
        
        # since the keys for the start and end states are the same, add an iteration tracker to make sure sure algo doesn't immediate return
        iter += 1
    
    return came_from, cost_so_far


def main(args):
    FUNCTION_MAP = {'naive_euclidean' : naive_euclidean,
                    'avg_remaining' : avg_remaining}

    parser = argparse.ArgumentParser()
    parser.add_argument('--heuristic', choices=FUNCTION_MAP.keys(), required=True, type=str)
    parser.add_argument('--problem', required=True, type=str)
    args = parser.parse_args()

    with open(args.problem, 'rb') as input_file:
        lines = input_file.readlines()

    # exits if input file is empty
    if len(lines) <= 1:
        print("Error: empty input")
        exit(1)

    # parse input file for list of cities
    cities = []
    for line in lines[1:]:
        x, y = line.split()
        cities.append( (int(x), int(y)) )
    start = cities[0]
    
    # start tracking execution time
    start_time = time.time()

    # create held-karp graph from list and run A* search, 
    a_star_graph = Graph(start, cities)
    came_from, cost_so_far = a_star_search(a_star_graph, (start, None), start, FUNCTION_MAP[args.heuristic])

    # comment out printing when testing for performance
    print("HELD-KARP GRAPH:")
    pprint.pprint(a_star_graph.__dict__)
    print()
    print("NODES EXPANDED:")
    pprint.pprint(came_from)
    print()
    print("INTERMEDIATE COSTS:")
    pprint.pprint(cost_so_far)
    print()

    print("FINAL COST:")
    final_keys = [x for x in cost_so_far.keys() if start in x]
    final_items = [cost_so_far[x] for x in final_keys]
    final_cost = max(final_items)
    print(final_cost)
    print()

    # print total execution time
    time_elapsed = time.time() - start_time 
    print_standardized_time(time_elapsed)

if __name__ == "__main__":
    main(sys.argv[1:])