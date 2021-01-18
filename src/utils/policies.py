from typing import Any, Callable, Sequence
import numpy as np
from numba import njit
from PyExpUtils.utils.arrays import argsmax
from PyExpUtils.utils.random import sample

class Policy:
    def __init__(self, probs: Callable[[Any], Sequence[float]], rng = np.random):
        self.probs = probs
        self.random = rng

    def selectAction(self, s: Any):
        action_probabilities = self.probs(s)
        return sample(action_probabilities)

    def ratio(self, other: Any, s: Any, a: int) -> float:
        probs = self.probs(s)
        return probs[a] / other.probs(s)[a]

def fromStateArray(probs: Sequence[Sequence[float]]):
    return Policy(lambda s: probs[s])

def fromActionArray(probs: Sequence[float]):
    return Policy(lambda s: probs)

@njit(cache=True)
def epsilonGreedy(qs: np.ndarray, epsilon: float):
    max_acts = argsmax(qs)
    actions = len(qs)
    pi = np.zeros(actions)

    for a in max_acts:
        pi[a] = 1. / len(max_acts)

    # compute a uniform random policy
    uniform = np.ones(actions) / actions

    # epsilon greedy is a mixture of greedy + uniform random
    return (1. - epsilon) * pi + epsilon * uniform

@njit(cache=True)
def softmax(preference_values : np.ndarray):
    t = np.exp(preference_values - np.max(preference_values))
    pi = t / np.sum(t)
    return pi

def createEpsilonGreedy(seed: int, epsilon: float, getValues: Callable[[Any], np.ndarray]):
    rng = np.random.RandomState(seed)
    policy = Policy(lambda s: epsilonGreedy(getValues(s), epsilon), rng)
    return policy

def createSoftmax(seed: int, getPreferences: Callable[[Any], np.ndarray]):
    rng = np.random.RandomState(seed)
    policy = Policy(lambda s : softmax(getPreferences(s)), rng)
    return policy
