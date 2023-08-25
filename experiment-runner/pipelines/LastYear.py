import numpy

from pipelines.LastWYears_Average import AvgLastWYears

def make_hyper_grid(**kwargs):
    return {
            'W': [1],
        }

def construct(*args, **kwargs):
    return AvgLastWYears(*args, **kwargs)