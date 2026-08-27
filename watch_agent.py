import time
from pathlib import Path

from bubbles import BubblePopEnv, clear_console


MODEL_DIR = Path(__file__).resolve().parent
MODEL_PREFIX = "bubble_pop_model"


def find_latest_model_path():
    best_model_paths = list(MODEL_DIR.glob(f"{MODEL_PREFIX}_*_best_*.zip"))

    if best_model_paths:
        return max(best_model_paths, key=lambda path: path.stat().st_mtime), True

    model_paths = list(MODEL_DIR.glob(f"{MODEL_PREFIX}*.zip"))

    if not model_paths:
        return None, False

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
    model_path, is_best_model = find_latest_model_path()

    if model_path is None:
        print("No trained model found. Run this first:")
        print("python train_agent.py")
        return

    if is_best_model:
        print("Loading best evaluated model:", model_path)
    else:
        print("Loading model:", model_path)
    model, uses_action_masks = load_model(model_path)

    if model is None:
        return

    env = BubblePopEnv()
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
