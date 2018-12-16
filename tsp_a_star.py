import sys
import argparse
import timeit
import math
import heapq
import itertools

# data structure for reading in file and converting to a graph
class Graph:
    # source and sink are equivalent to startpt (1 in this example)
    # node: [(neighbor, cost)]
    # {   
    #     ('source', {}): [ ((2, {2}), d(1,2)), ((3, {3}), d(1,3)), ((4, {4}), d(1,4)) ] 
    #     (2, {2}): [ ((3, {2,3}), d(2,3)), ((4, {2,4}), d(2,4)) ],
    #     ...,
    #     ((3, {2,3}): [ ((4, {2,3,4}), d(3,4)) ],
    #     ...,
    #     (4, {2,3,4}): [ ("sink", d(4,1)) ],
    # }
    ''' 
    (x, S): starting from 1, path min cost ends at vertex x, passing vertices in set S exactly once
    source and sink are equivalent to startpt
    {   
        ("source", None): [ (2, None), (3, None), (4, None) ] 
        (2, None): [ (3, (2)), (4, (2)) ],
        ...,
        (3, (2)): [ (4, (2,3)) ],
        ...,
        (4, (2,3)): [ ("sink", (2,3,4)) ],
    }

    '''
    # adjacency list for graph, takes a start point and a list of cities as input
    def __init__(self, startpt, cities):
        self.startpt = startpt
        self.edges = {}
        # add source node with key "source"
        cities = set(cities)
        cities.discard(startpt)
        # create all possible combinations of cities
        # [[(2,), (3,), (4,)], [(2, 3), (2, 4), (3, 4)], [(2, 3, 4)]]
        # city_combinations = [list(itertools.combinations(cities, n)) for n in range(1, len(cities) + 1)]
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
    
    def __repr__(self):
        return str(self.__dict__)

    def neighbors(self, point):
        return self.edges[point]

    def cost(self, from_node, to_node):
        from_point = from_node[0]
        to_point = to_node[0]
        if from_point == "source":
            return rectilinear(self.startpt, to_point)
        elif to_point == "sink":
            return rectilinear(from_point, self.startpt)
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

def held_karp(startpt, cities):
    cities = set(cities)
    cities.remove(startpt)
    cities = tuple(cities)

    # Quick and dirty memoization
    lookuptable = {}

    def min_dist(loc, path):
        """
        Minimum distance starting at city startpt, visiting all cities in path,
        and finishing at city loc
        """
        if (loc, path) not in lookuptable:

            if len(path) == 1 and loc in path:
                lookuptable[(loc, path)] = rectilinear(startpt, loc)
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

# used to estimate cost of going from a to b, must optimistic (less than the true cost)
def heuristic(a, b):
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
                priority = new_cost # + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far


def main(args):
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
    # print(a_star_graph)
    print(a_star_search(a_star_graph, ("source", None), "sink"))

if __name__ == "__main__":
    main(sys.argv[1:])