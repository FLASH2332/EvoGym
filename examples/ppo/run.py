import argparse
from typing import Optional
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ppo.eval import eval_policy
from ppo.callback import EvalCallback


def run_ppo(
    args: argparse.Namespace,
    body: np.ndarray,
    env_name_1: str,  # harish add two environments
    env_name_2: str,
    model_save_dir: str,
    model_save_name: str,
    connections: Optional[np.ndarray] = None,
    seed: int = 42,
) -> float:
    """
    Run ppo and return the best reward achieved during evaluation.
    """

    # Parallel environments
    vec_env_1 = make_vec_env(
        env_name_1,
        n_envs=1,
        seed=seed,
        env_kwargs={  # harish create two vec_env
            "body": body,
            "connections": connections,
        },
    )

    vec_env_2 = make_vec_env(
        env_name_2,
        n_envs=1,
        seed=seed,
        env_kwargs={
            "body": body,
            "connections": connections,
        },
    )

    # Eval Callback
    callback1 = EvalCallback(  # use two different callback and model
        body=body,
        connections=connections,
        env_name=env_name_1,
        eval_every=args.eval_interval,
        n_evals=args.n_evals,
        n_envs=args.n_eval_envs,
        model_save_dir=model_save_dir,
        model_save_name=model_save_name + f"_{env_name_1}",
        verbose=args.verbose_ppo,
    )

    # Train
    model1 = PPO(
        "MlpPolicy",
        vec_env_1,
        verbose=args.verbose_ppo,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
    )
    model1.learn(
        total_timesteps=args.total_timesteps,
        callback=callback1,
        log_interval=args.log_interval,
    )

    # Eval Callback
    callback2 = EvalCallback(  # use two different callback and model
        body=body,
        connections=connections,
        env_name=env_name_2,
        eval_every=args.eval_interval,
        n_evals=args.n_evals,
        n_envs=args.n_eval_envs,
        model_save_dir=model_save_dir,
        model_save_name=model_save_name + f"_{env_name_2}",
        verbose=args.verbose_ppo,
    )

    # Train
    model2 = PPO(
        "MlpPolicy",
        vec_env_2,
        verbose=args.verbose_ppo,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
    )
    model2.learn(
        total_timesteps=args.total_timesteps,
        callback=callback2,
        log_interval=args.log_interval,
    )

    dictFit = {"Walker-v0":10.99,"Carrier-v0":10.99}
    fitnessMax1 = dictFit[env_name_1]
    fitnessMax2 = dictFit[env_name_2]

    reward = callback1.best_reward/fitnessMax1 + callback2.best_reward/fitnessMax2 - abs(callback1.best_reward/fitnessMax1 - callback2.best_reward/fitnessMax2)

    return reward
