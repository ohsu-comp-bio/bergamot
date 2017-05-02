#!/usr/bin/env python

import sys
import csv
import gzip

if sys.argv[1].endswith(".gz"):
    reader = csv.reader(gzip.GzipFile(sys.argv[1]), delimiter="\t")
else:
    reader = csv.reader(open(sys.argv[1]), delimiter="\t")
header = None

if sys.argv[2] == "-":
    ohandle = sys.stdout
else:
    ohandle = open(sys.argv[2], "w")

writer = csv.writer(ohandle, delimiter="\t")

SKIP = ["3'UTR", "Silent", "5'UTR", "Intron"]

for row in reader:
    if header is None:
        header = row
        writer.writerow( row )
    else:
        o = []
        for a in row[1:]:
            v = list(b for b in a.split(",") if len(b) and b not in SKIP)
            o.append(len(v) > 0)
        #if sum(o) > 20:
        writer.writerow( [row[0]] + list(1 if a else 0 for a in o) )
