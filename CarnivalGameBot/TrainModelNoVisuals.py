import gymnasium as gym
import ale_py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Tworzenie środowiska Carnival-v5
def make_env():
    return gym.make("ALE/Carnival-v5")  # Usunięto render_mode='human' dla trenowania

# Tworzymy wektorowe środowisko dla stabilności trenowania
env = DummyVecEnv([make_env])

# Parametry PPO (Proximal Policy Optimization)
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_carnival_tensorboard/",
    learning_rate=0.0001,
    n_steps=2048,
    normalize_advantage=True
)

# Trenowanie modelu
model.learn(total_timesteps=50_000)  # Zmniejszona liczba timesteps dla szybszego uczenia

# Ewaluacja agenta
eval_results = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Średni wynik: {eval_results[0]} +/- {eval_results[1]}")

# Zapisujemy model
model.save("ppo_carnival")

# Możliwość wczytania modelu później
# model = PPO.load("ppo_carnival", env=env)

# Testowanie modelu i wizualizacja
def test_agent(model, env, episodes=5):
    env_render = DummyVecEnv([lambda: gym.make("ALE/Carnival-v5", render_mode='human')])  # Renderowanie tylko w testach
    for episode in range(episodes):
        obs = env_render.reset()
        done = False
        score = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env_render.step(action)
            score += reward
        print(f"Episode {episode + 1}: Score {score}")
    env_render.close()

test_agent(model, env)