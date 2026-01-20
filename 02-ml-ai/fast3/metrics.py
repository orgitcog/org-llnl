import math
import numpy as np
import torch

from collections import defaultdict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from scipy.special import xlogy
from pdb import set_trace
from utils.general_util import log
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn import functional as F
class Meter(object):

    def __init__(self, metrics, main_metric):
        self.metrics = metrics
        self.main_metric = main_metric
        self.initial_metric = 10E9 \
            if self.metrics[main_metric].better == "low" \
            else -10E9
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.labels = []
        self.results = {}

    def is_better(self, results, value):
        if self.metrics[self.main_metric].better == "low":
            return results[self.main_metric] < value
        else:
            return results[self.main_metric] > value

    def update(self, predictions, targets, predicted_labels=None, true_labels=None):
        predictions = predictions.cpu().float().detach().numpy()
        targets = targets.cpu().float().detach().numpy()
        if predicted_labels is not None:
            predicted_labels = predicted_labels.cpu().float().detach().numpy()
            true_labels = true_labels.cpu().float().detach().numpy()

        if predicted_labels is not None:
            self.labels.append(predicted_labels)


        self.predictions.append(predictions)
        self.targets.append(targets)

        results = {}
        for metric_name, metric_fn in self.metrics.items():
            if metric_name.startswith(('bce', 'ce')):
                results[metric_name] = metric_fn.calculate(predicted_labels, true_labels)
            else:

                results[metric_name] = metric_fn.calculate(predictions, targets)
        return results

    def compute(self):
        for metric_name, metric_fn in self.metrics.items():
            self.results[metric_name] = metric_fn.calculate(
                np.concatenate(self.predictions),
                np.concatenate(self.targets)
            )
        return self.results


class Metric(object):
    def __init__(self, better="low", kwargs={}):
        self.better = better
        self.kwargs = kwargs

    def calculate(self, predictions, targets):
        pass


class R2(Metric):
    def __init__(self, kwargs={}):
        super(R2, self).__init__("high", **kwargs)

    def calculate(self, predictions, targets):
        return r2_score(y_true=targets, y_pred=predictions, **self.kwargs)

class MAE(Metric):
    def calculate(self, predictions, targets):
        return mean_absolute_error(y_true=targets, y_pred=predictions, **self.kwargs)

class MSE(Metric):
    def calculate(self, predictions, targets):
        return mean_squared_error(y_true=targets, y_pred=predictions, **self.kwargs)

class RMSE(Metric):
    def calculate(self, predictions, targets):
        return math.sqrt(mean_squared_error(y_true=targets, y_pred=predictions, **self.kwargs))

class PearsonR(Metric):
    def __init__(self, kwargs={}):
        super(PearsonR, self).__init__("high", **kwargs)

    def calculate(self, predictions, targets):
        try:
            return stats.pearsonr(x=targets.reshape(-1), y=predictions.reshape(-1), **self.kwargs)[0]
        except:
            set_trace()

class SpearmanR(Metric):
    def __init__(self, kwargs={}):
        super(SpearmanR, self).__init__("high", **kwargs)

    def calculate(self, predictions, targets):
        return stats.spearmanr(targets.reshape(-1), predictions.reshape(-1), **self.kwargs)[0]


class BCE(Metric):
    def __init__(self, kwargs={}):
        super(BCE, self).__init__("low", **kwargs)
        self.criterion = BCEWithLogitsLoss()
    def calculate(self, predictions, targets):
        return float(self.criterion(torch.from_numpy(predictions), F.sigmoid(torch.from_numpy(targets))).cpu().detach().numpy())
    
class CE(Metric):
    def __init__(self, kwargs={}):
        super(CE, self).__init__("low", **kwargs)
        self.criterion = CrossEntropyLoss()
    def calculate(self, predictions, targets):
        return float(self.criterion(torch.from_numpy(predictions), torch.from_numpy(targets)).cpu().detach().numpy())