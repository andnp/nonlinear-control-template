from typing import Dict
import numpy as np
import torch
import torch.nn.functional as f

from agents.BaseAgent import BaseAgent
from utils.torch import device, Batch

# TODO: decide if you want to use mellowmax
# otherwise swap this for a simple max or weighted mean op for Q-learning
# or ESARSA respectively
def mellowmax(x: torch.Tensor, omega: float):
    c = x.max(1).values.unsqueeze(1)

    num = torch.mean(torch.exp(omega * (x - c)), 1).unsqueeze(1)
    return c + torch.log(num) / omega

class QLearning(BaseAgent):
    def forward(self, batch):
        q, = self.value_net(batch.states)
        qp, = self.target_net(batch.nterm_next_states)
        return {
            "value": q,
            "next_value": qp,
        }

    def bootstrap(self, batch: Batch, next_values: torch.Tensor):
        # if the whole batch is made of terminal states (which *does* happen)
        # then default to a bunch of zeros
        q_sp_ap = torch.zeros(batch.size, 1, device=device)
        if batch.nterm_next_states.shape[0] > 0:
            q_sp_ap[batch.is_non_terminals] = mellowmax(next_values, self.omega)

        target = batch.rewards + self.gamma * q_sp_ap

        return target, {
            "q_sp_ap": q_sp_ap
        }

    def updateNetwork(self, batch: Batch, predictions: Dict[str, torch.Tensor]):
        q_s = predictions['value']
        q_s_a = q_s.gather(1, batch.actions)

        target, _ = self.bootstrap(batch, predictions['next_value'])
        loss = 0.5 * f.mse_loss(target.detach(), q_s_a)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        delta = loss.mean().detach().cpu().numpy()
        return np.sqrt(delta)
