import time
import flappy_bird_gym
import joblib
from datetime import datetime
import numpy as np

training = False
episodes = 50000
fps = 30
report_freq = 100
save_freq = 5000

alpha = 0.7
alpha_decay = 0.00001
gamma = 0.95


# Q-table
# map (h_dist, v_dist, velocity) => [nothing, flap]

def init_q_table():
    try:
        q_table = joblib.load("q_table.pkl")
    except FileNotFoundError:
        q_table = dict()

    return q_table


def save_q_table(q_table):
    joblib.dump(q_table, "q_table.pkl")


def get_state(q_table, h_dist, v_dist, vel):
    # inspired by https://towardsdatascience.com/reinforcement-learning-in-python-with-flappy-bird-37eb01a4e786
    # which reduces the number of states and learns faster
    if h_dist < 140:
        h_dist = int(h_dist) - (int(h_dist) % 10)
    else:  # far away, reduce state
        h_dist = int(h_dist) - (int(h_dist) % 70)

    if -180 < v_dist < 180:
        v_dist = int(v_dist) - (int(v_dist) % 10)
    else:  # far away, reduce state
        v_dist = int(v_dist) - (int(v_dist) % 60)

    state = (int(h_dist), int(v_dist), int(vel))

    if state not in q_table:
        q_table[state] = [0.0, 0.0]

    return q_table, state


def render_frame(env, fps=30):
    env.render()
    time.sleep(1 / fps)


if __name__ == '__main__':
    env = flappy_bird_gym.make("FlappyBird-v0")

    q_table = init_q_table()

    max_score = 0

    for i in range(1, episodes + 1):
        moves = []

        obs = env.reset()
        q_table, state = get_state(q_table, *obs)

        while True:
            action = np.argmax(q_table[state])

            new_obs, _, done, info = env.step(action)
            q_table, new_state = get_state(q_table, *new_obs)

            moves.append((state, action, new_state))

            max_score = max(max_score, info["score"])
            state = new_state

            if not training:
                render_frame(env, fps)

            if training and done:  # update Q-table
                moves = list(reversed(moves))
                for j in range(0, len(moves)):
                    state, action, new_state = moves[j]
                    reward = -100 if j <= 1 else 0

                    q_table[state][action] = (1 - alpha) * (q_table[state][action]) + alpha * (
                            reward + gamma * max(q_table[new_state]))

                if alpha > 0.1:
                    alpha = max(alpha - alpha_decay, 0.1)

                break  # exit episode

        if i % report_freq == 0:
            print(
                f"({datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}) | Episode: {i} | States: {len(q_table)} | Max score: {max_score}"
            )

        if training and (i == episodes or i % save_freq == 0):
            save_q_table(q_table)

    env.close()
    print("Done...")
