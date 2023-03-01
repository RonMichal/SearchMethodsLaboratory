from random import randint
from astar import find_path
from math import sqrt
import pickle
import csv
from os import remove
from numpy.random import normal

MIN_COORDINATE = 0  # Minimum coordinate in the map
MAX_COORDINATE = 25  # Maximum coordinate in the map
MAX_HEIGHT = 10  # Maximum height in the map
BUILDING_WIDTH_MEAN = 5  # Mean of the X axis size of a building
BUILDING_WIDTH_STD = 0.5  # Standard deviation of the X axis size of a building
BUILDING_DEPTH_MEAN = 5  # Mean of the Y axis size of a building
BUILDING_DEPTH_STD = 0.5  # Standard deviation of the Y axis size of a building
BUILDING_HEIGHT_MEAN = 8  # Maximum a building can go in the Z axis
BUILDING_HEIGHT_STD = 2  # Standard deviation of the Z axis size of a building
MAX_ITERATIONS = 100000  # Maximum number of iterations to generate buildings
BUILDING_COUNTS = [125, 250, 375, 500, 625, 750, 875, 1000, 1125] #[500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]  # Number of buildings to generate
MAP_COUNT = 20  # Number of maps to generate
GOAL_PER_MAP = 3  # Number of goals to generate per map

def generate_random_buildings(number_of_buildings):
    # Function generates random rectangular buildings for the map
    # Input: number_of_buildings - number of buildings to generate
    # Output: list of buildings
    def buildings_intersect(building1, building2):
        # Function checks if two buildings intersect
        # Input: building1, building2 - lists of building parameters
        # Output: True if buildings intersect, False otherwise
        x1, y1, w1, d1, h1 = building1
        x2, y2, w2, d2, h2 = building2
        if (x1 + w1 < x2 or x2 + w2 < x1 or
                y1 + d1 < y2 or y2 + d2 < y1):
            return False
        return True

    def is_building_legal(building):
        # Function checks if a building is legal
        # Input: building - list of building parameters
        # Output: True if building is legal, False otherwise
        x, y, w, d, h = building
        if (x < MIN_COORDINATE * 5 or x + w > MAX_COORDINATE*5 or
                y < MIN_COORDINATE or y + d > MAX_COORDINATE or
                h > MAX_HEIGHT):
            return False
        return True

    def generate_building():
        # Function generates random building
        # Input: None
        # Output: list of building parameters
        x = randint(MIN_COORDINATE*5 + 1, MAX_COORDINATE*5)
        y = randint(MIN_COORDINATE + 1, MAX_COORDINATE)
        width = round(normal(BUILDING_WIDTH_MEAN, BUILDING_WIDTH_STD))
        depth = round(normal(BUILDING_DEPTH_MEAN, BUILDING_DEPTH_STD))
        height = round(normal(BUILDING_HEIGHT_MEAN, BUILDING_HEIGHT_STD))
        return [x, y, width, depth, height]

    buildings = []
    iteration = 0
    while len(buildings) < number_of_buildings and iteration < MAX_ITERATIONS:
        current_building = generate_building()
        if is_building_legal(current_building) and not any(
                buildings_intersect(current_building, building) for building in buildings):
            buildings.append(current_building)
        iteration += 1
    return buildings


def get_blocked_states(buildings):
    # Function generates blocked states for the map
    # Input: buildings - list of buildings
    # Output: set of blocked states
    blocked_states = set()
    for building in buildings:
        x, y, width, depth, height = building
        for i in range(width):
            for j in range(depth):
                for k in range(height):
                    blocked_states.add((x + i, y + j, k))
    return blocked_states


def generate_blocked_states(number_of_buildings):
    buildings = generate_random_buildings(number_of_buildings)
    return get_blocked_states(buildings)


def neighbor_function(blocked_states, state):
    def legal_coordinates(state):
        return MIN_COORDINATE <= state[0] <= MAX_COORDINATE and MIN_COORDINATE <= state[
            1] <= MAX_COORDINATE and MIN_COORDINATE <= state[2] <= MAX_HEIGHT

    if state in blocked_states:
        return []
    x, y, z = state
    candidate_neighbors = [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)]
    return [neighbor for neighbor in candidate_neighbors if
            neighbor not in blocked_states and legal_coordinates(neighbor)]


def neighbor_distance_function(state1, state2):
    x1, y1, z1 = state1
    x2, y2, z2 = state2
    if x1 == x2 and y1 == y2 and z1 - z2 == -1:  # Going down
        return 1.0
    return 1.5 + (z1 * (1 / 7))  # Other actions will cost more if the height is higher


def manhattan_distance(state1, state2):
    x1, y1, z1 = state1
    x2, y2, z2 = state2
    return abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)


def euclidean_distance(state1, state2):
    x1, y1, z1 = state1
    x2, y2, z2 = state2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def weighted_heuristic(state1, state2):
    return 2 * manhattan_distance(state1, state2)


def chebyshev_distance(state1, state2):
    x1, y1, z1 = state1
    x2, y2, z2 = state2
    return max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2))


def run_algorithms(building_count, blocked_states, goal, map_index, goal_index):
    for heuristic in [manhattan_distance, euclidean_distance, weighted_heuristic, chebyshev_distance]:
        print(f"Running algorithm with {heuristic.__name__} for goal {goal_index} in map {map_index}...")
        path, expanded, time, cost = find_path((0, 0, 0), goal, lambda state: neighbor_function(blocked_states, state),
                                               reversePath=True, heuristic_cost_estimate_fnct=heuristic,
                                               distance_between_fnct=neighbor_distance_function)
        # save the results to csv file
        with open('results.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([building_count, len(blocked_states), map_index, goal, heuristic.__name__, expanded, time, cost])


def save_map(buildings, map_index):
    city = [[' ' for _ in range(MAX_COORDINATE + 1)] for _ in range(MAX_COORDINATE + 1)]
    for building in buildings:
        x, y, width, depth, height = building
        for i in range(width):
            for j in range(depth):
                city[x + i][y + j] = str(height)
    with open(f"map{map_index}.p", "wb") as f:
        pickle.dump(city, f)

def draw_map(buildings, goal):
    city = [[' ' for _ in range(MAX_COORDINATE*5 + 1)] for _ in range(MAX_COORDINATE + 1)]
    for building in buildings:
        x, y, width, depth, height = building
        for i in range(width):
            for j in range(depth):
                city[x + i][y + j] = str(height)
    x, y, z = goal
    city[x][y] = 'g'
    print('\n'.join([''.join(y) for y in city]))
    
def get_goal(blocked_states, previous_goals_for_map):
    def not_in_first_quarter(candidate):
        x, y, z = candidate
        return x > MAX_COORDINATE / 2 or y > MAX_COORDINATE / 2

    while True:
        goal = (randint(0, MAX_COORDINATE), randint(0, MAX_COORDINATE), randint(0, MAX_HEIGHT))
        if goal not in blocked_states and goal not in previous_goals_for_map and not_in_first_quarter(goal):
            return goal


def main():
    delete_prev_results = input("Delete previous results? (y/n) ")
    if delete_prev_results.lower() == 'y':
        try:
            remove('results.csv')
        except:
            pass
    delete_prev_maps = input("Delete previous maps? (y/n) ")
    if delete_prev_maps.lower() == 'y':
        for map_index in range(1, MAP_COUNT):
            try:
                remove(f"map{map_index}.p")
            except:
                pass

    with open('results.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['buildings', 'blocked nodes', 'map', 'goal', 'heuristic', 'expanded', 'time', 'cost'])

    for building_count in BUILDING_COUNTS:
        for map_index in range(1, MAP_COUNT + 1):
            print(f"Generating map {map_index} for building count {building_count}...")

            # Generating new map
            buildings = generate_random_buildings(building_count)
            blocked_states = get_blocked_states(buildings)
            # save_map(buildings, map_index)

            # Running algorithm on goals
            previous_goals_for_map = set()
            for goal_index in range(1, GOAL_PER_MAP + 1):
                print(f"Generating goal {goal_index} for map {map_index}...")
                goal = get_goal(blocked_states, previous_goals_for_map)
                run_algorithms(len(buildings), blocked_states, goal, map_index, goal_index)
                previous_goals_for_map.add(goal)


if __name__ == "__main__":
    buildings = generate_random_buildings(30)
    draw_map(buildings,(25,25,0))
