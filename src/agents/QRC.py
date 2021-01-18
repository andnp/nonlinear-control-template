from typing import Dict, Optional
import numpy as np
import torch
import torch.nn.functional as f

from agents.TwoTimescale import TwoTimescale
from utils.torch import Batch

class QRC(TwoTimescale):
    def __init__(self, features, actions, params, seed, collector):
        super().__init__(features, actions, params, seed, collector)

        # define parameter contract
        self.beta:float = params.get('beta', 1.0)
        self.beta_rate:float = params.get('beta_rate', 0.99)
        self.original_beta = self.beta
        self.adaptive:Optional[bool] = params.get('adaptive_beta', True)

        self.delta_magnitude:float = 1.0

        self.updates: int = 0

    def updateBeta(self, delta):
        if not self.adaptive:
            return self.original_beta

        # track magnitude of delta
        self.delta_magnitude = self.beta_rate * self.delta_magnitude + (1 - self.beta_rate) * delta.square().mean().item()
        self.beta = np.min((self.original_beta, 1.0 / (self.delta_magnitude + 1e-8)))

        return self.beta

    def updateNetwork(self, batch: Batch, predictions: Dict[str, torch.Tensor]):
        self.updates += 1
        Qs = predictions['value']
        delta_hats = predictions['secondary']

        # collect the outputs based on which action was taken
        delta_hat = delta_hats.gather(1, batch.actions)
        Qsa = Qs.gather(1, batch.actions)

        # build the bootstrapped target
        target, aux = self.bootstrap(batch, predictions['next_value'])
        Qspap = aux['q_sp_ap'] # grab the auxiliary variables to avoid recomputing

        # compute the TD loss
        td_loss = 0.5 * f.mse_loss(target.detach(), Qsa)

        # compute the loss for the correction term head
        delta = target - Qsa
        h_loss = 0.5 * f.mse_loss(delta.detach(), delta_hat)

        beta = self.updateBeta(delta)

        regularizer = 0.5 * beta * torch.norm(self.h_weight)**2
        regularizer += 0.5 * beta * torch.norm(self.h_bias_weight)**2

        loss = td_loss + h_loss + regularizer + torch.mean(self.gamma * delta_hat.detach() * Qspap)

        self.optimizer.zero_grad()
        self.h_optimizer.zero_grad()
        self.target_net.zero_grad()

        loss.backward()

        # add the gradients of the target network for the correction term to the gradients for the td error
        self.combineTargetGrads()

        self.optimizer.step()
        self.h_optimizer.step()

        return delta.detach().squeeze(0).numpy()
