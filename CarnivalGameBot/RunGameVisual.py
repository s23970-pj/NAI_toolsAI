'''
Problem: Trening i testowanie agenta Reinforcement Learning w środowisku Carnival-v5 z Wizualizacją postępu
Autor: Adrian Goik, Łukasz Soldatke
Opis: Skrypt wykorzystuje algorytm PPO (Proximal Policy Optimization) do nauki gry w Carnival-v5
Przygotowanie środowiska:
zainstaluj poleceniem pip install [nazwa pakietu]
-gymnasium
-ale_py
-tensorflow
-numpy
-matplotlib.pyplot
-stable_baselines3
Instrukcja użycia:
1. Uruchom skrypt, aby wytrenować model PPO.
2. Możesz zapisać i ponownie wczytać model.
3. Skrypt zawiera funkcję testującą agenta i wizualizującą wyniki.
Referencje:
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- Gymnasium ALE Carnival-v5: https://gymnasium.farama.org/environments/atari/carnival/

UWAGA: PRZED URUCHOMIENIEM SKRYPTU W CELU TRENINGU WYCIĄGNIJ model.save z komentarza.
'''
import gymnasium as gym
import ale_py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


def make_env():
    """
    Tworzy i zwraca środowisko gry Carnival-v5.

    Returns:
        gym.Env: Obiekt środowiska Carnival-v5.
    """
    return gym.make("ALE/Carnival-v5", render_mode='human')


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

# # Trenowanie modelu
# model.learn(total_timesteps=50_000)  # Zmniejszona liczba timesteps dla szybszego uczenia


def test_agent(model, env, episodes=5):
    """
    Testuje wytrenowanego agenta w podanym środowisku przez określoną liczbę epizodów.

    Parameters:
        model (PPO): Wytrenowany model PPO.
        env (gym.Env): Środowisko gry Carnival-v5.
        episodes (int): Liczba epizodów do przetestowania (domyślnie 5).
    """
    for episode in range(episodes):
        obs = env.reset()
        done = False
        score = 0
        frames = []
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            score += reward
            frames.append(env.render(mode='rgb_array'))
        print(f"Episode {episode + 1}: Score {score}")

        # Wizualizacja rozgrywki
        plt.figure(figsize=(15, 5))
        for i, frame in enumerate(frames):
            plt.subplot(1, len(frames), i + 1)
            plt.imshow(frame)
            plt.axis("off")
        plt.show()


# Ewaluacja agenta
def evaluate_agent(model, env, n_eval_episodes=10):
    """
    Ewaluacja wytrenowanego modelu na określonej liczbie epizodów.

    Parameters:
        model (PPO): Wytrenowany model PPO.
        env (gym.Env): Środowisko gry Carnival-v5.
        n_eval_episodes (int): Liczba epizodów do ewaluacji (domyślnie 10).

    Returns:
        tuple: Średni wynik i odchylenie standardowe z ewaluacji.
    """
    eval_results = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    print(f"Średni wynik: {eval_results[0]} +/- {eval_results[1]}")
    return eval_results


# Uruchomienie ewaluacji
evaluate_agent(model, env)

# # Zapisujemy model
# model.save("ppo_carnival")

# Możliwość wczytania modelu później
model = PPO.load("ppo_carnival", env=env)

# Testowanie agenta
test_agent(model, env)
