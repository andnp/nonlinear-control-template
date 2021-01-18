from abc import abstractmethod
from typing import Dict
import torch
import copy

from agents.Network.Network import Network, cloneNetworkWeights
from agents.Network.serialize import deserializeOptimizer
from utils.ReplayBuffer import ReplayBuffer
from utils.per.per import PrioritizedReplayMemory
from utils.torch import addGradients_, device, getBatchColumns, Batch
from utils.policies import createEpsilonGreedy
from PyExpUtils.utils.Collector import Collector

class BaseAgent:
    def __init__(self, features: int, actions: int, params: Dict, seed: int, collector: Collector):
        self.features = features
        self.actions = actions
        self.params = params
        self.collector = collector
        self.seed = seed

        # define parameter contract
        self.gamma = params['gamma']
        self.epsilon = params.get('epsilon', 0)
        # the mellowmax parameter
        self.omega = params.get('omega', 1.0)

        # set up network for estimating Q(s, a)
        self.value_net = Network(features, actions, params, seed).to(device)

        # build the optimizer
        self.optimizer_params = params['optimizer']
        self.optimizer = deserializeOptimizer(self.value_net.parameters(), self.optimizer_params)

        self.steps = 0

        # set up the replay buffer
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch']
        self.buffer_type = params.get('buffer', 'standard')

        if self.buffer_type == 'per':
            prioritization = params['prioritization']
            self.buffer = PrioritizedReplayMemory(self.buffer_size, prioritization)
        else:
            self.buffer = ReplayBuffer(self.buffer_size)

        # build a target network
        self.target_refresh = params.get('target_refresh', 1)
        self.target_net = copy.deepcopy(self.value_net)
        self.initializeTargetNet()

        def getValues(x: torch.Tensor):
            qs = self.values(x).detach().cpu().squeeze(0).numpy()
            return qs

        self.policy = createEpsilonGreedy(seed, self.epsilon, getValues)

    # return the Q(s, a) values from the value network
    def values(self, x):
        return self.value_net(x)[0]

    # sample an action according to our policy
    def selectAction(self, x):
        return self.policy.selectAction(x)

    def initializeTargetNet(self):
        # if we aren't using target nets, then save some compute
        if self.target_refresh > 1:
            self.target_net = copy.deepcopy(self.value_net)
            cloneNetworkWeights(self.value_net, self.target_net)
        else:
            self.target_net = self.value_net

    @abstractmethod
    def updateNetwork(self, batch: Batch, predictions: Dict):
        pass

    @abstractmethod
    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def bootstrap(self, batch: Batch, next_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

    # a helper method that lets us bypass combining gradients whenever
    # target networks are disabled
    def combineTargetGrads(self):
        if self.target_net == self.value_net:
            return

        addGradients_(self.value_net, self.target_net)

    def update(self, s, a, sp, r, gamma):
        self.buffer.add((s, a, sp, r, gamma))
        self.steps += 1

        if self.steps % self.target_refresh == 0 and self.target_refresh > 1:
            cloneNetworkWeights(self.value_net, self.target_net)

        if len(self.buffer) > self.batch_size + 1:
            samples, idcs = self.buffer.sample(self.batch_size)
            batch = getBatchColumns(samples)
            predictions = self.forward(batch)
            tde = self.updateNetwork(batch, predictions)

            self.buffer.update_priorities(idcs, tde)
