import sys
import argparse
import timeit
import math
import heapq


# data structure for reading in file and converting to a graph
class Graph:
    ''' 
    # source and sink are equivalent to startpt (1 in this example)
    # node: [(neighbor, cost)]
    {   
        ('source', {}): [ ((2, {2}), d(1,2)), ((3, {3}), d(1,3)), ((4, {4}), d(1,4)) ] 
        (2, {2}): [ ((3, {2,3}), d(2,3)), ((4, {2,4}), d(2,4)) ],
        ...,
        ((3, {2,3}): [ ((4, {2,3,4}), d(3,4)) ],
        ...,
        (4, {2,3,4}): [ ("sink", d(4,1)) ],
    }
     '''
    # adjacency list for graph
    def __init__(self, startpt, memo_table):
        self.edges = {}
        # add source node with key "source"
        size = 1
        connected_points = [x for x in memo_table if x[0] in x[1] and len(x[1]) == size]
        connected_values = [(x, rectilinear(x[0], startpt)) for x in connected_points]
        self.edges[('source', {})] = connected_values
        # add rest of the nodes to graph
        while connected_points:
            tmp = []
            size += 1
            for point in connected_points:
                next_points = [x for x in memo_table if point[0] in x[1] and len(x[1]) == size]
                next_values = [(x, rectilinear(x[0], point[0])) for x in next_points]
                tmp += next_points
                # continue attaching nodes to previous nodes
                if tmp:
                    self.edges[point] = next_values
                    connected_points = tmp
                # full set reached, connect to sink node
                else:
                    self.edges[point] = [("sink", rectilinear(startpt, point[0]))]

    def neighbors(self, point):
        return self.edges[point]

    def cost(self, from_node, to_node):
        return rectilinear(from_node, to_node)

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
            new_cost = cost_so_far[current] + next[1]
            # update cost of next neighbor if applicable
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                # nodes with lower from_cost + heuristic_cost have higher priority (lower number)
                priority = new_cost + heuristic(goal, next)
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

    cities = []
    for line in lines[1:]:
        x, y = line.split()
        cities.append((x,y))
    
    start = cities[0]
    memo = held_karp(start, cities)
    a_star_graph = Graph(cities[0], memo)
    print(a_star_search(a_star_graph, ('source', {}), "sink"))

if __name__ == "__main__":
    main(sys.argv[1:])