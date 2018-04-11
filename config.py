
class Config:
    def __init__(self):
        self.max_steps = 3000000
        self.max_steps_per_episode = 300
        self.max_episodes = 15000
        self.gamma = 0.99
        self.state_size = [64, 64, 1]
        self.action_size = 5
        self.load_model = False
        self.model_path = './model'