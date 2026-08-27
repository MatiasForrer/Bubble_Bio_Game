import argparse
import re
import time
from pathlib import Path

from bubbles import (
    BubblePopEnv,
    board_move_interval as default_board_move_interval,
    clear_console,
    height_game_map,
    rounds,
    start_height as default_start_height,
    total_colours,
    width_game_map,
)


MODEL_DIR = Path(__file__).resolve().parent
MODEL_PREFIX = "bubble_pop_model"
MODEL_CONFIG_PATTERN = re.compile(
    r"_w(?P<width>\d+)"
    r"_h(?P<height>\d+)"
    r"_c(?P<total_colours>\d+)"
    r"_start(?P<start_height>\d+)"
    r"_rounds(?P<max_rounds>\d+)"
    r"_move(?P<board_move_interval>\d+)"
)
MODEL_AVG_PATTERN = re.compile(r"_avg(?P<average_score>\d+)\.zip$")


def parse_model_metadata(model_path):
    filename = model_path.name
    config_match = MODEL_CONFIG_PATTERN.search(filename)
    average_match = MODEL_AVG_PATTERN.search(filename)

    env_kwargs = {
        "width": width_game_map,
        "height": height_game_map,
        "total_colours": total_colours,
        "start_height": default_start_height,
        "max_rounds": rounds,
        "board_move_interval": default_board_move_interval,
    }

    if config_match is not None:
        env_kwargs.update({
            key: int(value)
            for key, value in config_match.groupdict().items()
        })

    average_score = None
    if average_match is not None:
        average_score = int(average_match.group("average_score"))

    return env_kwargs, average_score


def matches_requested_board(model_path, width, height):
    env_kwargs, average_score = parse_model_metadata(model_path)

    if width is not None and env_kwargs["width"] != width:
        return False

    if height is not None and env_kwargs["height"] != height:
        return False

    return True


def best_model_sort_key(model_path):
    env_kwargs, average_score = parse_model_metadata(model_path)

    if average_score is None:
        average_score = -1

    return average_score, model_path.stat().st_mtime


def find_latest_model_path(width=None, height=None):
    model_paths = [
        model_path
        for model_path in MODEL_DIR.glob(f"{MODEL_PREFIX}*.zip")
        if matches_requested_board(model_path, width, height)
    ]

    if not model_paths:
        return None, False

    best_model_paths = [
        model_path
        for model_path in model_paths
        if "_best_" in model_path.name
    ]

    if best_model_paths:
        return max(best_model_paths, key=best_model_sort_key), True

    return max(model_paths, key=lambda path: path.stat().st_mtime), False


def load_model(model_path):
    maskable_load_error = None

    try:
        from sb3_contrib import MaskablePPO

        return MaskablePPO.load(model_path), True
    except ImportError:
        pass
    except Exception as error:
        maskable_load_error = error

    try:
        from stable_baselines3 import PPO

        return PPO.load(model_path), False
    except ImportError:
        print("Install RL packages first:")
        print("pip install gymnasium stable-baselines3 sb3-contrib")
        return None, False
    except Exception as error:
        print("Could not load model:", model_path)
        if maskable_load_error is not None:
            print("MaskablePPO load failed:", maskable_load_error)
        print("PPO load failed:", error)
        return None, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()

    if args.model is not None:
        model_path = args.model
        is_best_model = "_best_" in model_path.name
    else:
        model_path, is_best_model = find_latest_model_path(args.width, args.height)

    if model_path is None:
        print("No trained model found. Run this first:")
        print("python train_agent.py")
        return

    if not model_path.exists():
        print("Model file does not exist:", model_path)
        return

    env_kwargs, average_score = parse_model_metadata(model_path)

    if is_best_model:
        print("Loading best evaluated model:", model_path)
    else:
        print("Loading model:", model_path)

    print(
        "Environment:",
        f"{env_kwargs['width']}x{env_kwargs['height']}",
        "| colours",
        env_kwargs["total_colours"],
        "| start height",
        env_kwargs["start_height"],
        "| max rounds",
        env_kwargs["max_rounds"],
        "| board move interval",
        env_kwargs["board_move_interval"]
    )

    if average_score is not None:
        print("Validation average from filename:", average_score)

    model, uses_action_masks = load_model(model_path)

    if model is None:
        return

    env = BubblePopEnv(**env_kwargs)
    observation, info = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        clear_console()
        print("score", env.score)
        print("next ball colour:", env.next_ball)
        print()
        env.render()

        if uses_action_masks:
            action, state = model.predict(
                observation,
                action_masks=env.get_action_mask(),
                deterministic=True
            )
        else:
            action, state = model.predict(observation, deterministic=True)

        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.4)

    clear_console()
    print("finished")
    print("score", env.score)
    print()
    env.render()


if __name__ == "__main__":
    main()
