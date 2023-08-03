# from source.agent import Agent
from source.environment import Environment


def main():

    env = Environment()

    # while 

    env.startGame()
    state, reward, done = env.step(action=0)
    print(f"{state=}, {reward=}, {done=}")

    env.reset()


if __name__ == "__main__":
    main()