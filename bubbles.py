import json
import os
import random
from pathlib import Path

import numpy as np


total_colours = 3
width_game_map = 10 # width of game_map
height_game_map = 12 # threshold for end of game_map
start_height = 3 # amount of populated rows from the top
rounds = 30
GAMEOVER = False
stagger_offset = 0
leaderboard_file = Path(__file__).with_name("leaderboard.json")

game_map = np.zeros([height_game_map, width_game_map], dtype=int)


def GAMEOVER_check(game_map):
    global GAMEOVER

    # If any ball reaches the bottom row, the game is over.
    if np.any(game_map[-1, :] != 0):
        GAMEOVER = True
        return game_map

    # The bottom row is still empty, so remove it before adding a new top row.
    GAMEOVER = False
    game_map = game_map[:-1, :]
    return game_map


def new_top_row(game_map):
    global stagger_offset

    width = game_map.shape[1]

    # A new row is one row high and as wide as the current game_map.
    new_row = np.array([[random.randint(1, total_colours) for x in range(width)]])
    game_map = np.vstack([new_row, game_map])

    # Adding a row at the top shifts the whole hex pattern down one row.
    stagger_offset = 1 - stagger_offset

    # Keep the grid height correct.
    game_map = game_map[:height_game_map, :]
    return game_map


def new_game(game_map):
    for i in range(start_height):
        game_map = new_top_row(game_map)
    return game_map


def row_is_shifted(y):
    # The offset changes when rows are added, so the hex pattern moves with the balls.
    return (y + stagger_offset) % 2 == 1


def get_neighbours(game_map, x, y):
    height, width = game_map.shape

    # Shifted and unshifted rows use different diagonals in a staggered hex grid.
    if not row_is_shifted(y):
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

        if 0 <= nx < width and 0 <= ny < height:
            neighbours.append((nx, ny))

    return neighbours


def pop(game_map, ball_x, ball_y):
    target_colour = game_map[ball_y, ball_x]

    if target_colour == 0:
        return game_map, 0, 0

    to_search = [(ball_x, ball_y)]
    visited = set()
    matching_balls = []

    # Flood-fill through connected balls of the same colour.
    while to_search:
        x, y = to_search.pop()

        if (x, y) in visited:
            continue

        visited.add((x, y))

        if game_map[y, x] == target_colour:
            matching_balls.append((x, y))

            for nx, ny in get_neighbours(game_map, x, y):
                if (nx, ny) not in visited:
                    to_search.append((nx, ny))

    # Groups of 3 or more disappear.
    popped_balls = 0
    floating_balls = 0

    if len(matching_balls) >= 3:
        popped_balls = len(matching_balls)

        for x, y in matching_balls:
            game_map[y, x] = 0

        game_map, floating_balls = remove_floating_balls(game_map)

    return game_map, popped_balls, floating_balls


def remove_floating_balls(game_map):
    height, width = game_map.shape
    attached_balls = set()
    to_search = []

    # Balls connected to the top row are still attached to the field.
    for x in range(width):
        if game_map[0, x] != 0:
            to_search.append((x, 0))

    # Find every ball connected to the top row through other balls.
    while to_search:
        x, y = to_search.pop()

        if (x, y) in attached_balls:
            continue

        attached_balls.add((x, y))

        for nx, ny in get_neighbours(game_map, x, y):
            if game_map[ny, nx] != 0 and (nx, ny) not in attached_balls:
                to_search.append((nx, ny))

    floating_balls = 0

    # Any ball not connected to the top row is floating, so remove it.
    for y in range(height):
        for x in range(width):
            if game_map[y, x] != 0 and (x, y) not in attached_balls:
                game_map[y, x] = 0
                floating_balls += 1

    return game_map, floating_balls


def is_playable(game_map, x, y):
    height, width = game_map.shape

    # The chosen position must be inside the grid and empty.
    if not (0 <= x < width and 0 <= y < height):
        return False

    if game_map[y, x] != 0:
        return False

    # Top-row empty spaces are playable because balls can attach there.
    if y == 0:
        return True

    # Other empty spaces are playable only if they touch an existing ball.
    for nx, ny in get_neighbours(game_map, x, y):
        if game_map[ny, nx] != 0:
            return True

    return False


def get_playable_positions(game_map):
    playable_positions = []
    height, width = game_map.shape

    for y in range(height):
        for x in range(width):
            if is_playable(game_map, x, y):
                playable_positions.append((x, y))

    return playable_positions


def map_pop(game_map, x, y, ball_colour):
    # Place the played ball, then pop connected balls from that position.
    game_map[y, x] = ball_colour
    game_map, popped_balls, floating_balls = pop(game_map, x, y)
    return game_map, popped_balls, floating_balls


def calculate_round_score(popped_balls, floating_balls, survived):
    # Normal pops, floating pops, and survival each add to the score.
    round_score = popped_balls * 10
    round_score += floating_balls * 15

    total_popped = popped_balls + floating_balls

    # Bigger clears get a simple combo bonus.
    if total_popped >= 8:
        round_score += 60
    elif total_popped >= 5:
        round_score += 25

    if survived:
        round_score += 5

    return round_score


def game_round(game_map, x, y, ball_colour):
    game_map, popped_balls, floating_balls = map_pop(game_map, x, y, ball_colour)
    game_map = GAMEOVER_check(game_map)

    # Only add a new top row if the bottom row did not end the game.
    survived = not GAMEOVER
    if survived:
        game_map = new_top_row(game_map)

    round_score = calculate_round_score(popped_balls, floating_balls, survived)
    return game_map, round_score, popped_balls, floating_balls


def clear_console():
    # Use the normal clear command for the player's operating system.
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")


def load_leaderboard():
    if not leaderboard_file.exists():
        return []

    try:
        with open(leaderboard_file, "r") as file:
            leaderboard = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []

    return leaderboard


def save_leaderboard(leaderboard):
    with open(leaderboard_file, "w") as file:
        json.dump(leaderboard, file, indent=4)


def update_leaderboard(player_name, score):
    leaderboard = load_leaderboard()
    leaderboard.append({"name": player_name, "score": score})

    # Highest scores should be shown first, and only the top 3 are kept.
    leaderboard.sort(key=lambda entry: entry["score"], reverse=True)
    leaderboard = leaderboard[:3]

    save_leaderboard(leaderboard)
    return leaderboard


def print_leaderboard(leaderboard):
    print("leaderboard")

    if not leaderboard:
        print("  no scores yet")
        return

    for index, entry in enumerate(leaderboard, start=1):
        print(f"  {index}. {entry['name']} - {entry['score']}")


def print_game_map(game_map):
    print()
    print("0 = empty, 1-" + str(total_colours) + " = ball colours")
    print("      " + "   ".join(str(x) for x in range(game_map.shape[1])))

    for y in range(game_map.shape[0]):
        # Use the same shifted-row rule as get_neighbours().
        row_indent = "  " if row_is_shifted(y) else ""
        row_values = "   ".join("." if value == 0 else str(value) for value in game_map[y])
        print(f"{y:2}: {row_indent}{row_values}")

    print()


def ask_for_position(playable_positions):
    while True:
        choice = input("Choose a position as x y: ").strip()
        parts = choice.split()

        if len(parts) != 2:
            print("Please enter two numbers, like: 3 4")
            continue

        try:
            x = int(parts[0])
            y = int(parts[1])
        except ValueError:
            print("Please enter numbers only.")
            continue

        if (x, y) in playable_positions:
            return x, y

        print("That position is not playable. Choose one from the list.")


def play_game():
    global game_map, GAMEOVER, stagger_offset

    GAMEOVER = False
    stagger_offset = 0
    score = 0
    last_round_message = ""
    game_map = np.zeros([height_game_map, width_game_map], dtype=int)
    game_map = new_game(game_map)
    current_round = 1

    while current_round <= rounds and not GAMEOVER:
        next_ball_colour = random.randint(1, total_colours)
        playable_positions = get_playable_positions(game_map)
        leaderboard = load_leaderboard()

        clear_console()
        print("round", current_round)
        print("score", score)
        if last_round_message:
            print(last_round_message)
        print_game_map(game_map)
        print("next ball colour:", next_ball_colour)
        print("playable positions:", playable_positions)
        print()
        print_leaderboard(leaderboard)
        print()

        if not playable_positions:
            print("no playable positions")
            break

        x, y = ask_for_position(playable_positions)
        game_map, round_score, popped_balls, floating_balls = game_round(game_map, x, y, next_ball_colour)
        score += round_score

        last_round_message = (
            "last round: +"
            + str(round_score)
            + " points, "
            + str(popped_balls)
            + " popped, "
            + str(floating_balls)
            + " floating popped"
        )
        current_round += 1

    clear_console()
    if GAMEOVER:
        print("game over")
    else:
        print("finished")

    print("final score:", score)
    print_game_map(game_map)
    player_name = input("Enter your name for the leaderboard: ").strip()

    if player_name == "":
        player_name = "player"

    leaderboard = update_leaderboard(player_name, score)
    print()
    print_leaderboard(leaderboard)


if __name__ == "__main__":
    play_game()
