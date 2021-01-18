from problems.CartPole import CartPole
from problems.MinBreakout import MinBreakout
from problems.MountainCar import MountainCar

def getProblem(name):
    if name == 'MountainCar':
        return MountainCar

    if name == 'CartPole':
        return CartPole

    if name == 'MinBreakout':
        return MinBreakout

    raise NotImplementedError()
