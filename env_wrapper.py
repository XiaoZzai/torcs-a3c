from torcs import Torcs
import cv2
import gym

class TorcsWrapper:
    def __init__(self, port=3101):
        self.episode = 0
        self.step_per_episode = 0
        self.step_total = 0

        self.env = Torcs(vision=True, port=port)

        # Discrete action space
        self.steers = [-0.10, -0.05, 0, 0.05, 0.10]
        self.action_space = gym.spaces.Discrete(5)

    def reset(self, track_offset=0):
        relaunch = False

        if self.episode % 3 == 0:
            relaunch = True

        self.episode += 1
        self.step_per_episode = 0

        ob= self.env.reset(relaunch=relaunch, track_offset=track_offset)
        img = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY) / 255.0
        return img.reshape(64, 64, 1)

    def step(self, steer_action):
        self.step_total += 1
        self.step_per_episode += 1
        ob, reward, done, _  = self.env.step(self.steers[steer_action])
        img = cv2.cvtColor(ob.img, cv2.COLOR_RGB2GRAY) / 255.0
        return img.reshape(64, 64, 1), reward, done

    def end(self):
        self.env.end()

if __name__ == "__main__":
    env = TorcsWrapper()
    img = env.reset()
    while True:
        cv2.imshow("img", img)
        cv2.waitKey(10)
        img, _, done = env.step(2)
        if done == True:
            break
    env.end()