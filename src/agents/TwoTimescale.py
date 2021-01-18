from typing import Dict
import torch

from agents.BaseAgent import deserializeOptimizer
from agents.QLearning import QLearning
from PyExpUtils.utils.Collector import Collector
from utils.torch import Batch

from PyExpUtils.utils.dict import merge

class TwoTimescale(QLearning):
    def __init__(self, features: int, actions: int, params: Dict, seed: int, collector: Collector):
        super().__init__(features, actions, params, seed, collector)

        self.h_grad:bool = self.params.get('h_grad', False)

        self.h = self.value_net.addOutput(actions, grad=self.h_grad, bias=True, initial_value=0)
        # re-copy this since we changed the policy net
        self.initializeTargetNet()

        # create a second optimizer specific to h
        alpha = self.optimizer_params['alpha']
        eta = self.optimizer_params.get('eta', 1.0)
        h_optimizer_params = merge(self.optimizer_params, { 'alpha': alpha * eta })

        learnables = self.h.parameters()
        self.h_optimizer = deserializeOptimizer(learnables, h_optimizer_params)

        self.h_weight = self.h.weight
        self.h_bias_weight = self.h.bias

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        q, secondary = self.value_net(batch.states)
        qp, _ = self.target_net(batch.nterm_next_states)

        return {
            "value": q,
            "secondary": secondary,
            "next_value": qp,
        }
