from sb3_contrib import TRPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os

def train_trpo(env_fn, total_timesteps=100000, model_path="trpo_model", eval_env_fn=None):
    env = make_vec_env(env_fn, n_envs=1)

    eval_callback = None
    if eval_env_fn:
        eval_env = make_vec_env(eval_env_fn, n_envs=1)
        eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                                     log_path="./logs/", eval_freq=10000)

    model = TRPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model
