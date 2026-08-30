import json
import os
import random
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


total_colours = 3
width_game_map = 8 # maximum row width; shifted rows use one fewer cell
height_game_map = 10 # threshold for end of game_map
start_height = 2 # amount of populated rows from the top
rounds = 40
board_move_interval = 2
leaderboard_file = Path(__file__).with_name("leaderboard.json")
INVALID_CELL = -1
MAX_BOUNCES = 3
CELL_BLOCK_RADIUS = 0.55
RAY_SAMPLE_STEP = 0.25


class SimpleDiscrete:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randrange(self.n)


class SimpleBox:
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


class SimpleDict:
    def __init__(self, spaces_dict):
        self.spaces = spaces_dict


class SimpleSpaces:
    Discrete = SimpleDiscrete
    Box = SimpleBox
    Dict = SimpleDict


if spaces is None:
    spaces = SimpleSpaces()


BaseEnv = gym.Env if gym is not None else object


def row_is_shifted(y, stagger_offset):
    # The offset changes when rows are added, so the hex pattern moves with the balls.
    return (y + stagger_offset) % 2 == 0


def row_real_width(y, max_width, stagger_offset):
    # Shifted rows are the shorter rows in the 9/10/9/10 pattern.
    if row_is_shifted(y, stagger_offset):
        return max_width - 1

    return max_width


def is_valid_cell(game_map, x, y, stagger_offset):
    height, width = game_map.shape

    if not (0 <= y < height):
        return False

    return 0 <= x < row_real_width(y, width, stagger_offset)


def apply_invalid_cells(game_map, stagger_offset):
    height, width = game_map.shape

    for y in range(height):
        real_width = row_real_width(y, width, stagger_offset)
        game_map[y, real_width:] = INVALID_CELL

    return game_map


def make_empty_game_map(height, width, stagger_offset):
    game_map = np.zeros((height, width), dtype=np.int64)
    return apply_invalid_cells(game_map, stagger_offset)


def get_neighbours(game_map, x, y, stagger_offset):
    if not is_valid_cell(game_map, x, y, stagger_offset):
        return []

    # Shifted and unshifted rows use different diagonals in a staggered hex grid.
    if not row_is_shifted(y, stagger_offset):
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

        if is_valid_cell(game_map, nx, ny, stagger_offset):
            neighbours.append((nx, ny))

    return neighbours


def add_new_top_row(game_map, total_colours, rng, max_height, stagger_offset):
    width = game_map.shape[1]
    new_stagger_offset = 1 - stagger_offset
    new_row_width = row_real_width(0, width, new_stagger_offset)

    # The environment controls randomness for new rows, not the agent.
    new_row = np.full((1, width), INVALID_CELL, dtype=np.int64)
    new_row[0, :new_row_width] = rng.integers(
        1,
        total_colours + 1,
        size=new_row_width,
        dtype=np.int64
    )
    game_map = np.vstack([new_row, game_map])

    # Adding a row at the top shifts the whole hex pattern down one row.
    stagger_offset = new_stagger_offset

    # Keep the grid height correct.
    game_map = game_map[:max_height, :]
    game_map = apply_invalid_cells(game_map, stagger_offset)
    return game_map, stagger_offset


def bottom_row_has_ball(game_map):
    return np.any(game_map[-1, :] > 0)


def remove_bottom_row(game_map):
    return game_map[:-1, :]


def remove_floating_balls(game_map, stagger_offset):
    height, width = game_map.shape
    attached_balls = set()
    to_search = []

    # Balls connected to the top row are still attached to the field.
    for x in range(row_real_width(0, width, stagger_offset)):
        if game_map[0, x] > 0:
            to_search.append((x, 0))

    # Find every ball connected to the top row through other balls.
    while to_search:
        x, y = to_search.pop()

        if (x, y) in attached_balls:
            continue

        attached_balls.add((x, y))

        for nx, ny in get_neighbours(game_map, x, y, stagger_offset):
            if game_map[ny, nx] > 0 and (nx, ny) not in attached_balls:
                to_search.append((nx, ny))

    floating_balls = 0

    # Any ball not connected to the top row is floating, so remove it.
    for y in range(height):
        for x in range(row_real_width(y, width, stagger_offset)):
            if game_map[y, x] > 0 and (x, y) not in attached_balls:
                game_map[y, x] = 0
                floating_balls += 1

    return game_map, floating_balls


def pop(game_map, ball_x, ball_y, stagger_offset):
    if not is_valid_cell(game_map, ball_x, ball_y, stagger_offset):
        return game_map, 0, 0

    target_colour = game_map[ball_y, ball_x]

    if target_colour <= 0:
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

            for nx, ny in get_neighbours(game_map, x, y, stagger_offset):
                if (nx, ny) not in visited:
                    to_search.append((nx, ny))

    popped_balls = 0
    floating_balls = 0

    # Groups of 3 or more disappear.
    if len(matching_balls) >= 3:
        popped_balls = len(matching_balls)

        for x, y in matching_balls:
            game_map[y, x] = 0

        game_map, floating_balls = remove_floating_balls(game_map, stagger_offset)

    return game_map, popped_balls, floating_balls


def cell_center(x, y, stagger_offset):
    # Shifted rows are drawn half a cell to the right.
    center_x = float(x)
    if row_is_shifted(y, stagger_offset):
        center_x += 0.5

    return center_x, float(y)


def shooter_position(game_map):
    height, width = game_map.shape

    # The shooter sits just below the board, centered between the side walls.
    return (width - 1) / 2, height + 0.75


def fold_x_into_board(x, left_wall, right_wall):
    # This mirrors x back between the walls after a virtual bounce.
    board_width = right_wall - left_wall
    folded = (x - left_wall) % (board_width * 2)

    if folded <= board_width:
        return left_wall + folded

    return right_wall - (folded - board_width)


def mirrored_target_x(target_x, left_wall, right_wall, mirror_index):
    # In the unfolded view, each neighbouring board represents one more wall bounce.
    board_width = right_wall - left_wall
    target_from_left = target_x - left_wall

    if mirror_index % 2 == 0:
        mirrored_from_left = mirror_index * board_width + target_from_left
    else:
        mirrored_from_left = (mirror_index + 1) * board_width - target_from_left

    return left_wall + mirrored_from_left


def get_occupied_centers(game_map, stagger_offset):
    occupied_centers = []
    height, width = game_map.shape

    for y in range(height):
        for x in range(row_real_width(y, width, stagger_offset)):
            if game_map[y, x] > 0:
                occupied_centers.append(cell_center(x, y, stagger_offset))

    return np.array(occupied_centers, dtype=np.float64)


def path_is_clear(game_map, target_x, target_y, mirror_index, stagger_offset, occupied_centers=None):
    shooter_x, shooter_y = shooter_position(game_map)
    left_wall = -0.5
    right_wall = game_map.shape[1] - 0.5
    target_center_x, target_center_y = cell_center(target_x, target_y, stagger_offset)
    mirror_x = mirrored_target_x(target_center_x, left_wall, right_wall, mirror_index)
    distance = np.hypot(mirror_x - shooter_x, target_center_y - shooter_y)
    samples = max(6, int(distance / RAY_SAMPLE_STEP))
    if occupied_centers is None:
        occupied_centers = get_occupied_centers(game_map, stagger_offset)

    if len(occupied_centers) == 0:
        return True

    block_radius_squared = CELL_BLOCK_RADIUS * CELL_BLOCK_RADIUS

    # Sample along the ray. Occupied bubbles are blockers before the target cell.
    for sample in range(1, samples):
        t = sample / samples
        ray_x = shooter_x + (mirror_x - shooter_x) * t
        ray_y = shooter_y + (target_center_y - shooter_y) * t
        folded_x = fold_x_into_board(ray_x, left_wall, right_wall)

        distances_squared = (occupied_centers[:, 0] - folded_x) ** 2
        distances_squared += (occupied_centers[:, 1] - ray_y) ** 2

        if np.any(distances_squared < block_radius_squared):
            return False

    return True


def has_clear_shot(game_map, x, y, stagger_offset, max_bounces=MAX_BOUNCES, occupied_centers=None):
    # Try a straight shot plus reflected shots off the left and right walls.
    for mirror_index in range(-max_bounces, max_bounces + 1):
        if path_is_clear(game_map, x, y, mirror_index, stagger_offset, occupied_centers):
            return True

    return False


def touches_existing_ball_or_ceiling(game_map, x, y, stagger_offset):
    if y == 0:
        return True

    for nx, ny in get_neighbours(game_map, x, y, stagger_offset):
        if game_map[ny, nx] > 0:
            return True

    return False


def is_playable(game_map, x, y, stagger_offset, occupied_centers=None):
    # The chosen position must be inside the grid and empty.
    if not is_valid_cell(game_map, x, y, stagger_offset):
        return False

    if game_map[y, x] != 0:
        return False

    if not touches_existing_ball_or_ceiling(game_map, x, y, stagger_offset):
        return False

    # Empty cells behind blockers are not playable, even if they touch a bubble.
    return has_clear_shot(game_map, x, y, stagger_offset, occupied_centers=occupied_centers)


def get_playable_positions(game_map, stagger_offset):
    playable_positions = []
    height, width = game_map.shape
    occupied_centers = get_occupied_centers(game_map, stagger_offset)

    for y in range(height):
        for x in range(row_real_width(y, width, stagger_offset)):
            if is_playable(game_map, x, y, stagger_offset, occupied_centers):
                playable_positions.append((x, y))

    return playable_positions


def calculate_round_score(popped_balls, floating_balls, survived):
    # Human score: clear scoring that rewards pops, floating clears, and survival.
    round_score = popped_balls * 10
    round_score += floating_balls * 15

    total_popped = popped_balls + floating_balls

    if total_popped >= 8:
        round_score += 60
    elif total_popped >= 5:
        round_score += 25

    if survived:
        round_score += 5

    return round_score


def calculate_rl_reward(popped_balls, floating_balls, survived, game_over):
    # RL reward: smaller numbers than score, with a small turn cost.
    reward = -0.5 * 2
    reward += popped_balls * 1.0 * 2
    reward += floating_balls * 1.5 * 2

    total_popped = popped_balls + floating_balls

    if total_popped >= 8:
        reward += 6.0
    elif total_popped >= 5:
        reward += 2.5

    if survived:
        reward += 0.5 * 3

    if game_over:
        reward -= 25.0

    return reward


def format_game_map(game_map, stagger_offset):
    lines = []
    width = game_map.shape[1]
    lines.append(
        "0 = empty, 1-"
        + str(total_colours)
        + f" = ball colours, rows alternate {width - 1}/{width} cells"
    )
    lines.append("      " + "   ".join(str(x) for x in range(game_map.shape[1])))

    for y in range(game_map.shape[0]):
        # Use the same shifted-row rule as get_neighbours().
        row_indent = "  " if row_is_shifted(y, stagger_offset) else ""
        real_width = row_real_width(y, game_map.shape[1], stagger_offset)
        row_values = "   ".join("." if value == 0 else str(value) for value in game_map[y, :real_width])
        lines.append(f"{y:2}: {row_indent}{row_values}")

    return "\n".join(lines)


class BubblePopEnv(BaseEnv):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        width=width_game_map,
        height=height_game_map,
        total_colours=total_colours,
        start_height=start_height,
        max_rounds=rounds,
        board_move_interval=board_move_interval,
        invalid_action_penalty=-5.0,
        seed=None
    ):
        if gym is not None:
            super().__init__()

        self.width = width
        self.height = height
        self.total_colours = total_colours
        self.start_height = start_height
        self.max_rounds = max_rounds
        self.board_move_interval = board_move_interval
        self.invalid_action_penalty = invalid_action_penalty
        self.rng = np.random.default_rng(seed)

        # The action is one board cell: action = y * width + x.
        self.action_space = spaces.Discrete(width * height)

        # Dictionary observations keep the board, next ball, and hex offset explicit.
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(INVALID_CELL, total_colours, shape=(height, width), dtype=np.int64),
            "next_ball": spaces.Box(1, total_colours, shape=(1,), dtype=np.int64),
            "stagger_offset": spaces.Box(0, 1, shape=(1,), dtype=np.int64),
        })

        self.game_map = make_empty_game_map(height, width, 0)
        self.next_ball = 1
        self.stagger_offset = 0
        self.score = 0
        self.round_number = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.game_map = make_empty_game_map(self.height, self.width, 0)
        self.stagger_offset = 0
        self.score = 0
        self.round_number = 0
        self.game_over = False

        for i in range(self.start_height):
            self.game_map, self.stagger_offset = add_new_top_row(
                self.game_map,
                self.total_colours,
                self.rng,
                self.height,
                self.stagger_offset
            )

        self.next_ball = self._random_ball_colour()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        x = int(action) % self.width
        y = int(action) // self.width

        if not is_playable(self.game_map, x, y, self.stagger_offset):
            info = self._get_info()
            info["invalid_action"] = True
            return self._get_observation(), self.invalid_action_penalty, False, False, info

        # Place the current ball, then pop connected balls from that position.
        self.game_map[y, x] = self.next_ball
        self.game_map, popped_balls, floating_balls = pop(
            self.game_map,
            x,
            y,
            self.stagger_offset
        )

        self.game_over = bottom_row_has_ball(self.game_map)
        survived = not self.game_over
        round_score = calculate_round_score(popped_balls, floating_balls, survived)
        self.score += round_score

        reward = calculate_rl_reward(popped_balls, floating_balls, survived, self.game_over)
        self.round_number += 1
        truncated = self.max_rounds is not None and self.round_number >= self.max_rounds
        board_moved = (
            survived
            and self.board_move_interval > 0
            and self.round_number % self.board_move_interval == 0
        )

        # The board only scrolls down every second valid move by default.
        if board_moved:
            self.game_map = remove_bottom_row(self.game_map)
            self.game_map, self.stagger_offset = add_new_top_row(
                self.game_map,
                self.total_colours,
                self.rng,
                self.height,
                self.stagger_offset
            )

        self.next_ball = self._random_ball_colour()

        # MaskablePPO needs at least one valid action. If none exist, end the episode.
        playable_positions_after = self.get_playable_positions()
        no_playable_moves = len(playable_positions_after) == 0
        if no_playable_moves:
            self.game_over = True
            reward -= 25.0

        info = self._get_info(playable_positions_after)
        info["invalid_action"] = False
        info["x"] = x
        info["y"] = y
        info["popped_balls"] = popped_balls
        info["floating_balls"] = floating_balls
        info["round_score"] = round_score
        info["no_playable_moves"] = no_playable_moves
        info["board_moved"] = board_moved

        return self._get_observation(), reward, self.game_over, truncated, info

    def render(self):
        output = format_game_map(self.game_map, self.stagger_offset)
        print(output)
        return output

    def get_action_mask(self):
        mask = np.zeros(self.width * self.height, dtype=bool)

        for x, y in get_playable_positions(self.game_map, self.stagger_offset):
            action = y * self.width + x
            mask[action] = True

        return mask

    def get_playable_positions(self):
        return get_playable_positions(self.game_map, self.stagger_offset)

    def _random_ball_colour(self):
        return int(self.rng.integers(1, self.total_colours + 1))

    def _get_observation(self):
        return {
            "grid": self.game_map.copy(),
            "next_ball": np.array([self.next_ball], dtype=np.int64),
            "stagger_offset": np.array([self.stagger_offset], dtype=np.int64),
        }

    def _get_info(self, playable_positions=None):
        if playable_positions is None:
            playable_positions = self.get_playable_positions()

        return {
            "score": self.score,
            "round_number": self.round_number,
            "next_ball": self.next_ball,
            "stagger_offset": self.stagger_offset,
            "playable_positions": playable_positions,
        }


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
    env = BubblePopEnv()
    env.reset()
    last_round_message = ""

    while env.round_number < rounds and not env.game_over:
        playable_positions = env.get_playable_positions()
        leaderboard = load_leaderboard()

        clear_console()
        print("round", env.round_number + 1)
        print("score", env.score)
        if last_round_message:
            print(last_round_message)
        print()
        env.render()
        print()
        print("next ball colour:", env.next_ball)
        print("playable positions:", playable_positions)
        print()
        print_leaderboard(leaderboard)
        print()

        if not playable_positions:
            print("no playable positions")
            break

        x, y = ask_for_position(playable_positions)
        action = y * env.width + x
        observation, reward, terminated, truncated, info = env.step(action)

        last_round_message = (
            "last round: +"
            + str(info["round_score"])
            + " points, "
            + str(info["popped_balls"])
            + " popped, "
            + str(info["floating_balls"])
            + " floating popped"
        )

        if info["board_moved"]:
            last_round_message += ", board moved down"
        else:
            last_round_message += ", board stayed"

        if terminated or truncated:
            break

    clear_console()
    if env.game_over:
        print("game over")
    else:
        print("finished")

    print("final score:", env.score)
    print()
    env.render()
    print()
    player_name = input("Enter your name for the leaderboard: ").strip()

    if player_name == "":
        player_name = "player"

    leaderboard = update_leaderboard(player_name, env.score)
    print()
    print_leaderboard(leaderboard)


if __name__ == "__main__":
    play_game()
