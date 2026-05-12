import random
import numpy as np
from time import sleep
from IPython.display import clear_output

total_colours = 3
width_map = 10 #width of game_map
height_map = 10 #threshold for end of game_map
start_height = 4 #amount of populated rows from the top
GAMEOVER = False

rounds =  20

game_map = np.zeros([height_map, width_map], dtype= int)

def GAMEOVER_check(game_map): #check if game has ended and removes bottom row
    global GAMEOVER
    for i in range(width_map):
        if game_map[height_map -1 ][i] != 0:
            print("game over")
            GAMEOVER = True
            break
    game_map = game_map[:height_map, :]
    # print("deleted")
    return game_map



def new_top_row(game_map):
    new_row = []
    for i in range(width_map):
        new_row.append(random.randint(1, total_colours))
        # print("assigned")
    game_map = np.vstack([new_row, game_map])
    return game_map



def new_game(game_map):
    for i in range(start_height - 1):
        game_map = new_top_row(game_map)
    return game_map

def map_pop(game_map):
    playable_positions = get_playable_positions(game_map)

    if not playable_positions:
        return game_map

    ball_x, ball_y = random.choice(playable_positions)

    game_map = pop(game_map, ball_x, ball_y)

    return game_map

def get_neighbours(x, y):
    if y % 2 == 0:
        directions = [
            (-1, 0),  # left
            (1, 0),   # right
            (-1, -1), # upper-left
            (0, -1),  # upper-right
            (-1, 1),  # lower-left
            (0, 1)    # lower-right
        ]
    else:
        directions = [
            (-1, 0),  # left
            (1, 0),   # right
            (0, -1),  # upper-left
            (1, -1),  # upper-right
            (0, 1),   # lower-left
            (1, 1)    # lower-right
        ]

    neighbours = []

    for dx, dy in directions:
        nx = x + dx
        ny = y + dy

        if 0 <= nx < width_map and 0 <= ny < height_map:
            neighbours.append((nx, ny))

    return neighbours

def pop(game_map, ball_x, ball_y):
    target_colour = game_map[ball_y][ball_x]

    if target_colour == 0:
        return game_map

    to_search = [(ball_x, ball_y)]
    visited = set()
    matching_balls = []

    while to_search:
        x, y = to_search.pop()

        if (x, y) in visited:
            continue

        visited.add((x, y))

        if game_map[y][x] == target_colour:
            matching_balls.append((x, y))

            for nx, ny in get_neighbours(x, y):
                if (nx, ny) not in visited:
                    to_search.append((nx, ny))

    if len(matching_balls) >= 3:
        for x, y in matching_balls:
            game_map[y][x] = 0

    return game_map


def game_round(game_map):
    game_map = map_pop(game_map)
    game_map = new_top_row(game_map)
    game_map = GAMEOVER_check(game_map)
    return game_map

def is_playable(game_map, x, y):
    # Must be inside the grid
    if not (0 <= x < width_map and 0 <= y < height_map):
        return False

    # Must be empty
    if game_map[y][x] != 0:
        return False

    # Top row is playable, because bubbles can attach there
    if y == 0:
        return True

    # Otherwise, it must touch an existing bubble
    for nx, ny in get_neighbours(x, y):
        if game_map[ny][nx] != 0:
            return True

    return False

def get_playable_positions(game_map):
    playable_positions = []

    for y in range(game_map.shape[0]):
        for x in range(game_map.shape[1]):
            if is_playable(game_map, x, y):
                playable_positions.append((x, y))

    return playable_positions

current_round = 0
while current_round < rounds and not GAMEOVER:
    if current_round == 0:
        game_map =  new_game(game_map)
    game_map = game_round(game_map)
    print(game_map)
    current_round += 1



