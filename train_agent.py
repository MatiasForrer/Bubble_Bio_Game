from pathlib import Path

from bubbles import BubblePopEnv


model_path = Path("bubble_pop_model")


def action_mask(env):
    # MaskablePPO uses this to block illegal board positions.
    return env.get_action_mask()


def train_with_maskable_ppo(total_timesteps):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    env = BubblePopEnv()
    env = ActionMasker(env, action_mask)

    # MultiInputPolicy is used because the observation is a dictionary.
    model = MaskablePPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)


def train_with_regular_ppo(total_timesteps):
    from stable_baselines3 import PPO

    env = BubblePopEnv()

    # Regular PPO can choose invalid actions, so the environment gives those a penalty.
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)


def main():
    total_timesteps = 10000

    try:
        train_with_maskable_ppo(total_timesteps)
        print("trained MaskablePPO model:", model_path)
    except ImportError:
        try:
            train_with_regular_ppo(total_timesteps)
            print("trained PPO model:", model_path)
        except ImportError:
            print("Install RL packages first:")
            print("pip install gymnasium stable-baselines3 sb3-contrib")


if __name__ == "__main__":
    main()
