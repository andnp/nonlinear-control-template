import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd() + '/src')

from PyExpUtils.results.results import loadResults
from analysis.results import findExpPath, getBest
from PyExpPlotting.learning_curves import plot
from analysis.colors import colors
from experiment.tools import parseCmdLineArgs
from experiment import ExperimentModel
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpPlotting.tools import getCurveReducer, findExperiments
from analysis.matplotlib import paramsToStr

setDefaultConference('jmlr')

ALG_ORDER = [ 'QRC' ]

def generatePlot(ax, exp_paths, bestBy):
    reducer = getCurveReducer(bestBy)
    for alg in ALG_ORDER:
        exp_path = findExpPath(exp_paths, alg)
        exp = ExperimentModel.load(exp_path)
        results = loadResults(exp, 'returns.csv')

        best = getBest(results, reducer)
        plot(best, ax, {
            'label': alg,
            'color': colors[alg],
            'width': 0.75,
        })


if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    exps = findExperiments(key='{domain}')
    for domain in exps:
        print(domain)

        if domain != 'CartPole':
            continue

        for bestBy in ['auc', 'end']:
            f, axes = plt.subplots(1)
            generatePlot(axes, exps[domain], bestBy)

            params = { 'bestBy': bestBy }

            file_name = f'{domain}_{paramsToStr(params)}'
            if should_save:
                save(
                    save_path=f'{path}/plots',
                    plot_name=file_name,
                    save_type=save_type,
                    width=0.3, # so we can stack these 3 wide
                    f=f,
                )
            else:
                plt.show()
                exit()
