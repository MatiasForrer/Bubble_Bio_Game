import time
from pathlib import Path

from bubbles import BubblePopEnv, clear_console


model_path = Path("bubble_pop_model")


def load_model():
    try:
        from sb3_contrib import MaskablePPO

        return MaskablePPO.load(model_path), True
    except ImportError:
        pass
    except FileNotFoundError:
        raise

    try:
        from stable_baselines3 import PPO

        return PPO.load(model_path), False
    except ImportError:
        print("Install RL packages first:")
        print("pip install gymnasium stable-baselines3 sb3-contrib")
        return None, False


def main():
    if not model_path.with_suffix(".zip").exists():
        print("No trained model found. Run this first:")
        print("python train_agent.py")
        return

    model, uses_action_masks = load_model()

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
