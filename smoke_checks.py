import numpy as np

from bubbles import (
    BubblePopEnv,
    bottom_row_has_ball,
    get_neighbours,
    has_clear_shot,
    is_playable,
    make_empty_game_map,
    pop,
    row_is_shifted,
    row_real_width,
)


def count_balls(game_map):
    return int(np.sum(game_map > 0))


def check_hex_stagger_offset():
    # When a new row is added, the old row moves down but keeps its visual shift.
    old_row_shift = row_is_shifted(0, 0)
    same_row_after_scroll = row_is_shifted(1, 1)

    assert old_row_shift == same_row_after_scroll
    assert row_real_width(0, 10, 0) == row_real_width(1, 10, 1)


def check_alternating_row_widths_and_neighbours():
    game_map = make_empty_game_map(4, 10, 0)

    assert row_real_width(0, 10, 0) == 9
    assert row_real_width(1, 10, 0) == 10
    assert row_real_width(2, 10, 0) == 9
    assert row_real_width(3, 10, 0) == 10

    neighbours = get_neighbours(game_map, 8, 0, 0)

    # The short row has no x=9 cell, so neighbours must not include it.
    assert (9, 0) not in neighbours


def check_pop_group():
    game_map = make_empty_game_map(3, 4, 1)
    game_map[0, 0] = 1
    game_map[0, 1] = 1
    game_map[0, 2] = 1

    game_map, popped_balls, floating_balls = pop(game_map, 1, 0, 1)

    assert popped_balls == 3
    assert floating_balls == 0
    assert count_balls(game_map) == 0


def check_floating_balls():
    game_map = make_empty_game_map(3, 4, 1)
    game_map[0, 0] = 1
    game_map[0, 1] = 1
    game_map[0, 2] = 1
    game_map[1, 1] = 2

    game_map, popped_balls, floating_balls = pop(game_map, 1, 0, 1)

    assert popped_balls == 3
    assert floating_balls == 1
    assert count_balls(game_map) == 0


def check_invalid_action():
    env = BubblePopEnv(seed=1)
    observation, info = env.reset()

    # Choose an invalid padding cell from a short row.
    invalid_y = None
    invalid_x = None
    for y in range(env.height):
        real_width = row_real_width(y, env.width, env.stagger_offset)
        if real_width < env.width:
            invalid_y = y
            invalid_x = real_width
            break

    invalid_action = invalid_y * env.width + invalid_x
    observation, reward, terminated, truncated, info = env.step(invalid_action)

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


def check_board_moves_every_second_move():
    env = BubblePopEnv(seed=3)
    observation, info = env.reset()
    starting_offset = env.stagger_offset

    first_action = int(np.flatnonzero(env.get_action_mask())[0])
    observation, reward, terminated, truncated, info = env.step(first_action)

    assert info["board_moved"] is False
    assert env.stagger_offset == starting_offset

    second_action = int(np.flatnonzero(env.get_action_mask())[0])
    observation, reward, terminated, truncated, info = env.step(second_action)

    assert info["board_moved"] is True
    assert env.stagger_offset != starting_offset


def check_game_over_detection():
    game_map = make_empty_game_map(4, 4, 0)
    game_map[-1, 0] = 1

    assert bottom_row_has_ball(game_map)


def check_shooter_visibility():
    game_map = make_empty_game_map(6, 10, 0)

    assert is_playable(game_map, 4, 0, 0)

    # A filled row blocks straight and bounce paths to cells hidden above it.
    for x in range(row_real_width(1, 10, 0)):
        game_map[1, x] = 2

    assert not has_clear_shot(game_map, 4, 0, 0)
    assert not is_playable(game_map, 4, 0, 0)


def main():
    check_hex_stagger_offset()
    check_alternating_row_widths_and_neighbours()
    check_pop_group()
    check_floating_balls()
    check_invalid_action()
    check_board_shape_stays_constant()
    check_board_moves_every_second_move()
    check_game_over_detection()
    check_shooter_visibility()
    print("all smoke checks passed")


if __name__ == "__main__":
    main()
