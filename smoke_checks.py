import numpy as np

from bubbles import (
    BubblePopEnv,
    bottom_row_has_ball,
    get_neighbours,
    pop,
    row_is_shifted,
)


def check_hex_stagger_offset():
    # When a new row is added, the old row moves down but keeps its visual shift.
    old_row_shift = row_is_shifted(0, 0)
    same_row_after_scroll = row_is_shifted(1, 1)

    assert old_row_shift == same_row_after_scroll

    normal_neighbours = get_neighbours(np.zeros((3, 3), dtype=int), 1, 1, 0)
    shifted_neighbours = get_neighbours(np.zeros((3, 3), dtype=int), 1, 1, 1)

    assert normal_neighbours != shifted_neighbours


def check_pop_group():
    game_map = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=int)

    game_map, popped_balls, floating_balls = pop(game_map, 1, 0, 0)

    assert popped_balls == 3
    assert floating_balls == 0
    assert np.sum(game_map) == 0


def check_floating_balls():
    game_map = np.array([
        [1, 1, 1],
        [0, 2, 0],
        [0, 0, 0],
    ], dtype=int)

    game_map, popped_balls, floating_balls = pop(game_map, 1, 0, 0)

    assert popped_balls == 3
    assert floating_balls == 1
    assert np.sum(game_map) == 0


def check_invalid_action():
    env = BubblePopEnv(seed=1)
    observation, info = env.reset()

    # Top rows are filled at reset, so action 0 is occupied and invalid.
    observation, reward, terminated, truncated, info = env.step(0)

    assert info["invalid_action"] is True
    assert reward < 0
    assert terminated is False


def check_board_shape_stays_constant():
    env = BubblePopEnv(seed=2)
    observation, info = env.reset()
    before_shape = env.game_map.shape
    action = int(np.flatnonzero(env.get_action_mask())[0])

    observation, reward, terminated, truncated, info = env.step(action)

    assert env.game_map.shape == before_shape


def check_game_over_detection():
    game_map = np.zeros((4, 4), dtype=int)
    game_map[-1, 0] = 1

    assert bottom_row_has_ball(game_map)


def main():
    check_hex_stagger_offset()
    check_pop_group()
    check_floating_balls()
    check_invalid_action()
    check_board_shape_stays_constant()
    check_game_over_detection()
    print("all smoke checks passed")


if __name__ == "__main__":
    main()
