from src.problems.BaseProblem import BaseProblem
from src.environments.Gym import Gym
from PyFixedReps.BaseRepresentation import BaseRepresentation

class IdentityRep(BaseRepresentation):
    def encode(self, s, a=None):
        return s

class MountainCar(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.actions = 3

        self.rep = IdentityRep()

        self.features = 2
        self.gamma = 0.99

        self.max_episode_steps = 1000

        # trick gym into thinking the max number of steps is a bit longer
        # that way we get to control the termination at max steps
        self.env = Gym('MountainCar-v0', self.run, self.max_episode_steps + 2)
