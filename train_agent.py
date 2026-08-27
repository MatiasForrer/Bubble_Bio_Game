import argparse
from datetime import datetime
from pathlib import Path

from bubbles import BubblePopEnv


MODEL_DIR = Path(__file__).resolve().parent
MODEL_PREFIX = "bubble_pop_model"


def action_mask(env):
    # MaskablePPO uses this to block illegal board positions.
    return env.get_action_mask()


def build_model_path(total_timesteps, n_steps, tag=None, completed_timesteps=None, average_score=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts = [
        MODEL_PREFIX,
        timestamp,
        f"ts{total_timesteps}",
        f"nsteps{n_steps}",
    ]

    if tag is not None:
        name_parts.append(tag)

    if completed_timesteps is not None:
        name_parts.append(f"step{completed_timesteps}")

    if average_score is not None:
        name_parts.append(f"avg{round(average_score)}")

    filename = "_".join(name_parts) + ".zip"
    return MODEL_DIR / filename


def evaluate_model_score(model, uses_action_masks, eval_episodes, seed_start):
    scores = []
    round_counts = []

    for episode_index in range(eval_episodes):
        seed = seed_start + episode_index
        env = BubblePopEnv(seed=seed)
        observation, info = env.reset(seed=seed)
        terminated = False
        truncated = False
        step_count = 0
        max_steps = env.max_rounds * 5 if env.max_rounds is not None else 200

        while not terminated and not truncated and step_count < max_steps:
            if uses_action_masks:
                action_mask = env.get_action_mask()

                if not action_mask.any():
                    break

                action, state = model.predict(
                    observation,
                    action_masks=action_mask,
                    deterministic=True
                )
            else:
                action, state = model.predict(observation, deterministic=True)

            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1

        scores.append(env.score)
        round_counts.append(env.round_number)

    average_score = sum(scores) / len(scores)
    average_rounds = sum(round_counts) / len(round_counts)
    return average_score, average_rounds, max(scores)


def make_score_evaluation_callback(
    total_timesteps,
    n_steps,
    eval_freq,
    eval_episodes,
    eval_seed_start,
    uses_action_masks
):
    if eval_freq <= 0 or eval_episodes <= 0:
        return None

    from stable_baselines3.common.callbacks import BaseCallback

    class ScoreEvaluationCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best_average_score = float("-inf")
            self.best_model_path = None
            self.next_eval_timestep = eval_freq
            self.last_eval_timestep = 0

        def _on_step(self):
            if self.num_timesteps >= self.next_eval_timestep:
                self._run_evaluation()

                while self.next_eval_timestep <= self.num_timesteps:
                    self.next_eval_timestep += eval_freq

            return True

        def _on_training_end(self):
            if self.last_eval_timestep != self.num_timesteps:
                self._run_evaluation()

        def _run_evaluation(self):
            average_score, average_rounds, best_score = evaluate_model_score(
                self.model,
                uses_action_masks,
                eval_episodes,
                eval_seed_start
            )
            self.last_eval_timestep = self.num_timesteps

            print(
                "eval step",
                self.num_timesteps,
                "| avg score",
                round(average_score, 2),
                "| avg rounds",
                round(average_rounds, 2),
                "| best episode",
                best_score
            )

            if average_score > self.best_average_score:
                self.best_average_score = average_score
                self.best_model_path = build_model_path(
                    total_timesteps,
                    n_steps,
                    tag="best",
                    completed_timesteps=self.num_timesteps,
                    average_score=average_score
                )
                self.model.save(self.best_model_path)
                print("saved new best model:", self.best_model_path)

    return ScoreEvaluationCallback()


def train_with_maskable_ppo(
    model_path,
    total_timesteps,
    n_steps,
    batch_size,
    n_epochs,
    learning_rate,
    device,
    eval_freq,
    eval_episodes,
    eval_seed_start
):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    env = BubblePopEnv()
    env = ActionMasker(env, action_mask)
    callback = make_score_evaluation_callback(
        total_timesteps,
        n_steps,
        eval_freq,
        eval_episodes,
        eval_seed_start,
        uses_action_masks=True
    )

    # MultiInputPolicy is used because the observation is a dictionary.
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=device,
        verbose=1
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(model_path)
    return callback


def train_with_regular_ppo(
    model_path,
    total_timesteps,
    n_steps,
    batch_size,
    n_epochs,
    learning_rate,
    device,
    eval_freq,
    eval_episodes,
    eval_seed_start
):
    from stable_baselines3 import PPO

    env = BubblePopEnv()
    callback = make_score_evaluation_callback(
        total_timesteps,
        n_steps,
        eval_freq,
        eval_episodes,
        eval_seed_start,
        uses_action_masks=False
    )

    # Regular PPO can choose invalid actions, so the environment gives those a penalty.
    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=device,
        verbose=1
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(model_path)
    return callback


def print_training_summary(algorithm_name, final_model_path, callback):
    print("trained", algorithm_name, "final model:", final_model_path)

    if callback is not None and callback.best_model_path is not None:
        print("best evaluated model:", callback.best_model_path)
        print("best average eval score:", round(callback.best_average_score, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=25000)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-freq", type=int, default=25000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed-start", type=int, default=1000)
    args = parser.parse_args()
    model_path = build_model_path(args.timesteps, args.n_steps)

    try:
        callback = train_with_maskable_ppo(
            model_path,
            args.timesteps,
            args.n_steps,
            args.batch_size,
            args.n_epochs,
            args.learning_rate,
            args.device,
            args.eval_freq,
            args.eval_episodes,
            args.eval_seed_start
        )
        print_training_summary("MaskablePPO", model_path, callback)
    except ImportError:
        try:
            callback = train_with_regular_ppo(
                model_path,
                args.timesteps,
                args.n_steps,
                args.batch_size,
                args.n_epochs,
                args.learning_rate,
                args.device,
                args.eval_freq,
                args.eval_episodes,
                args.eval_seed_start
            )
            print_training_summary("PPO", model_path, callback)
        except ImportError:
            print("Install RL packages first:")
            print("pip install gymnasium stable-baselines3 sb3-contrib")


if __name__ == "__main__":
    main()
