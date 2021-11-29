import sys
import numpy as np
import random
import math

# Calculate distance between cities
def distance(coords, city_0, city_1):
    coord_0, coord_1 = coords[city_0], coords[city_1]
    return math.sqrt((coord_0[0] - coord_1[0]) ** 2 + (coord_0[1] - coord_1[1]) ** 2)

# Calculate total distance of a given path
def pathDistance(solution, coords):
    n_cities = len(coords)
    result = 0
    for i in range(n_cities):
        result += distance(coords, solution[i % n_cities], solution[(i + 1) % n_cities])
    return result

# Generates an initial solution (traverse cities in order)
def initialSolution(coords):
    cities = [i for i in range(len(coords))]
    solution = []

    unvisited_cities = set(cities)
    while unvisited_cities:
        next_city = unvisited_cities.pop()
        solution.append(next_city)
        
    cur_pathDistance = pathDistance(solution, coords)
    return solution, cur_pathDistance

def generate_random_coords(num_citites):
    coords = []
    for i in range(num_citites):
        line = [random.uniform(-100, 100), random.uniform(-100, 100)]
        coords.append(np.array(line))
    return coords

def read_coords(path):
    coords = []
    try:
        with open(path, "r") as f:
            for line in f.readlines():
                line = [float(x.replace("\n", "")) for x in line.split(" ")]
                coords.append(np.array(line))
    except:
        print("** No se pudo leer el archivo", path)
        print("Se usaran 50 coordenadas generadas al azar")
        return generate_random_coords(50)
    return coords

# Global initializations
if len(sys.argv) < 2:
    print("Please indicate the filename with the coords or random # to generate the # coords")
    exit(1)
if(sys.argv[1] == 'random'):
    if len(sys.argv) < 3:
        print("Please enter the # of coords to generate after random")
        exit(1)
    coords = np.array(generate_random_coords(int(sys.argv[2])))
else:
    coords = np.array(read_coords(sys.argv[1]))

initialSolution, initialPathDistance = initialSolution(coords)
print("Initial solution path: ", initialSolution)
print("Initial path distance: ", initialPathDistance)
