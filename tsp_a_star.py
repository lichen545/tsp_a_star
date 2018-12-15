import sys
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
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

# implemented using a priority queue (pop from node, add neighbors of node to frontier, repeat)
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
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                # nodes with lower from_cost + heuristic_cost have higher priority (lower number)
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far

def main():
    return

if __name__ == "__main__":
    main()