# tsp_a_star
A implementation of the traveling salesman problem solved via A* search

New TSP problems can be generated via: `python generate_problem.py [# of cities desired]`
* example: `python generate_problem.py 3`
* example output: `tsp3.txt`

The A* search algorithm can be run via: `python tsp_a_star.py --problem [problem_file] --heuristic [selected_heuristic]`
* `[problem_file]` should be a generated txt file, e.g. `tsp3.txt`
* `[selected_heuristic]` should be a string selected from one of the following 2 options:
  * `naive_euclidean`
  * `avg_remaining`
* example: `python tsp_a_star.py --problem tsp3.txt --heuristic naive_euclidean`
  
