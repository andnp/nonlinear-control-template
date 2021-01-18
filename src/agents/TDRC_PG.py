import numpy as np
import torch
import torch.nn.functional as f

from agents.Network.Network import Network
from agents.Network.serialize import deserializeOptimizer
from agents.QRC import QRC
from utils.torch import Batch, device
from utils.policies import createSoftmax

class TDRC_PG(QRC):
    def __init__(self, features, actions, params, seed, collector):
        super().__init__(features, 1, params, seed, collector)
        # Define a policy net
        self.policy_net = Network(features, actions, params, seed).to(device)

        # Add the optimizer for policy
        self.policy_optimizer = deserializeOptimizer(self.policy_net.parameters(), self.optimizer_params)

        def getPreferences(x : torch.Tensor):
            prefs = self.action_preferences(x).detach().cpu().squeeze(0).numpy()
            return prefs

        self.policy = createSoftmax(seed, getPreferences)

    def action_preferences(self, x):
        return self.policy_net(x)[0]

    def updateNetwork(self, batch: Batch, predictions):
        Vs = predictions['value']

        prefs, = self.policy_net(batch.states)

        # compute log probabilities for policy gradient
        logsoftmax = torch.nn.LogSoftmax(dim = 1)
        log_pis = logsoftmax(prefs)
        log_pi = log_pis.gather(1, batch.actions)

        G, _ = self.bootstrap(batch, predictions['next_value'])
        delta = G - Vs

        p_loss = -torch.mean(delta.detach() * log_pi)

        self.policy_optimizer.zero_grad()
        p_loss.backward()

        # update the value functions according to QRC's update
        # but first lie to QRC that there's only one action
        tdrc_batch = Batch(
            batch.states,
            batch.nterm_next_states,
            torch.zeros_like(batch.actions),
            batch.rewards,
            batch.is_terminals,
            batch.is_non_terminals,
            batch.size,
        )
        ret = super().updateNetwork(tdrc_batch, predictions)
        # then finish updating the policy (in this order to make sure updates are all together)
        self.policy_optimizer.step()

        return ret
