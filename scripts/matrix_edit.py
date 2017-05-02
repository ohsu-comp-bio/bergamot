#!/usr/bin/env python

import os
import sys
import csv
import re
import math
import argparse

def value_eval(code, value):
    funcmap = {
        "len":len,
        "value" : value,
        "re" : re,
        "math" : math,
        "float" : float
    }
    return str(eval(code,{"__builtins__":None},funcmap))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--col-eval', help='Column Eval', dest="col_eval", default=None)
    parser.add_argument('-r', '--row-eval', help='Row Eval', dest="row_eval", default=None)
    parser.add_argument('-m', '--cell-eval', help='Cell Eval', dest="cell_eval", default=None)
    
    parser.add_argument("-o", "--out", help="Output File", dest="output", default=None)
    parser.add_argument("input", help="Input Matrix", default=None)
    
    args = parser.parse_args()

    if args.input == "-":
        ihandle = sys.stdin
    else:
        ihandle = open(args.input)

    if args.output is None:
        ohandle = sys.stdout
    else:
        ohandle = open(args.output, "w")

    reader = csv.reader(ihandle, delimiter="\t")
    writer = csv.writer(ohandle, delimiter="\t", lineterminator="\n")

    header = True
    for row in reader:
        if header:
            if args.col_eval is not None and len(args.col_eval):
                for i, val in enumerate(row[1:]):
                    row[i+1] = value_eval(args.col_eval, val)
            header = False
        else:
            if args.row_eval is not None and len(args.row_eval):
                row[0] = value_eval(args.row_eval, row[0])
          	if args.cell_eval is not None and len(args.cell_eval):
          		for i in range(1,len(row)):
          			row[i] = value_eval(args.cell_eval,row[i])    
        writer.writerow(row)

    ihandle.close()
    ohandle.close()


