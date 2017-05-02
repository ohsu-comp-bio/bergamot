#!/usr/bin/env python
import sys
import pandas
import numpy
from scipy.stats import expon
if __name__ == "__main__":
    matrix = pandas.read_csv(sys.argv[1], sep="\t", index_col=0)
    matrix = matrix.apply(lambda x: expon.ppf( (x.rank()-1) / len(x) ), 0)
    matrix = matrix.fillna(0.0)
    matrix.to_csv(sys.stdout, sep="\t")
