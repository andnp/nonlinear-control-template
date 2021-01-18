import torch
import numpy as np
from RlGlue import BaseAgent
from utils.torch import device

# keeps a one step memory for TD based agents
class OneStepWrapper(BaseAgent):
    def __init__(self, agent, gamma, rep):
        self.agent = agent
        self.gamma = gamma
        self.rep = rep

        self.s = None
        self.a = None
        self.x = None

    def start(self, s):
        self.s = s
        x = self.rep.encode(s)
        self.x = torch.tensor(x, device=device, dtype=torch.float32).unsqueeze(0)

        a = self.agent.selectAction(self.x)
        self.a = torch.tensor(a, device=device).unsqueeze(0)

        return a

    def step(self, r, sp):
        xp = self.rep.encode(sp)
        xp = torch.tensor(xp, device=device, dtype=torch.float32).unsqueeze(0)

        r = torch.tensor(r, device=device, dtype=torch.float32).view(1, 1)

        self.agent.update(self.x, self.a, xp, r, self.gamma)
        ap = self.agent.selectAction(xp)

        self.a = torch.tensor(ap, device=device).unsqueeze(0)

        self.s = sp
        self.x = xp

        return ap

    def end(self, r):
        r = torch.tensor(r, device=device, dtype=torch.float32).view(1, 1)
        self.agent.update(self.x, self.a, None, r, 0)
