import Box2D # we need to import this first because cedar is stupid
import numpy as np
import logging
import socket
import torch
import time
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from experiment import ExperimentModel
from problems.registry import getProblem
from PyExpUtils.utils.Collector import Collector
from utils.rlglue import OneStepWrapper

if len(sys.argv) < 2:
    print('run again with:')
    print('python3 src/main.py <path/to/description.json> <idx>')
    exit(1)

# try to detect if we are on a Cedar server
prod = 'cdr' in socket.gethostname()
# if we are local, then turn on info level logging
# otherwise, keep the console clear to improve performance and decrease clutter
if not prod:
    logging.basicConfig(level=logging.INFO)

torch.set_num_threads(1)

exp = ExperimentModel.load(sys.argv[1])
idx = int(sys.argv[2])

max_steps = exp.max_steps
run = exp.getRun(idx)

collector = Collector()

# set random seeds accordingly
np.random.seed(run)
torch.manual_seed(run)

Problem = getProblem(exp.problem)
problem = Problem(exp, idx)

agent = problem.getAgent(collector)
env = problem.getEnvironment()

wrapper = OneStepWrapper(agent, problem.getGamma(), problem.rep)

glue = RlGlue(wrapper, env)

# Run the experiment
glue.start()
start_time = time.time()
episode = 0
for step in range(exp.max_steps):
    _, _, _, t = glue.step()

    # on the terminal state or max steps
    # go back to start state
    # (gamma is only set to 0 on termination, not on max steps)
    is_episode_max = problem.max_episode_steps != -1 and glue.num_steps >= problem.max_episode_steps
    if t or is_episode_max:
        episode += 1
        glue.start()

        # collect an array of rewards that is the length of the number of steps in episode
        # effectively we count the whole episode as having received the same final reward
        collector.concat('returns', [glue.total_reward] * glue.num_steps)
        collector.collect('episodes', glue.total_reward)

        # compute the average time-per-step in ms
        avg_time = 1000 * (time.time() - start_time) / step
        logging.info(f' {episode} {step} {glue.total_reward} {avg_time:.4}ms')

        glue.total_reward = 0
        glue.num_steps = 0

collector.fillRest('returns', exp.max_steps)


# import matplotlib.pyplot as plt
# fig, ax1 = plt.subplots(1)

# return_data = collector.run_data['returns']
# ax1.plot(return_data)
# ax1.set_title('Return')

# plt.show()
# exit()

from PyExpUtils.results.backends.csv import saveResults
from PyExpUtils.utils.arrays import downsample

# only save the returns by default
# save auxiliary variables only when requested
to_save = ['returns', 'episodes']
if exp.save_aux:
    to_save = collector.run_data.keys()


for key in to_save:
    data = collector.run_data[key]

    # heavily downsample the data to reduce storage costs
    # we don't need all of the data-points for plotting anyways
    # method='window' returns a window average
    # method='subsample' returns evenly spaced samples from array
    # num=1000 makes sure final array is of length 1000
    # percent=0.1 makes sure final array is 10% of the original length (only one of `num` or `percent` can be specified)
    data = downsample(data, num=500, method='window')

    saveResults(exp, idx, key, data, precision=2)
