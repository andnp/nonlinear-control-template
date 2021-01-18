import os
import sys
from typing import Callable, Sequence
import numpy as np
sys.path.append(os.getcwd())

import PyExpUtils.utils.path as Path
from PyExpUtils.results.backends.backend import ResultList
from PyExpUtils.utils.arrays import first

def getBest(results: ResultList, reducer: Callable = np.mean):
    best = first(results)

    for r in results:
        a = r.mean()
        b = best.mean()
        am = reducer(a)
        bm = reducer(b)
        if am > bm:
            best = r

    return best

def findExpPath(arr: Sequence[str], alg: str):
    for exp_path in arr:
        if f'{alg.lower()}.json' == Path.fileName(exp_path.lower()):
            return exp_path

    raise Exception(f'Expected to find exp_path for {alg}')
