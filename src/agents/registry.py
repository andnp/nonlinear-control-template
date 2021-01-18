from agents.QRC import QRC
from agents.QLearning import QLearning
from agents.TDRC_PG import TDRC_PG

def getAgent(name):
    if name == 'QLearning':
        return QLearning

    if name == 'QRC':
        return QRC

    if name == 'TDRC-PG':
        return TDRC_PG

    raise NotImplementedError()
