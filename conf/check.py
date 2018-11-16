import sys

d = {}
for line in sys.stdin:
    ts = line.split('\t')
    d[len(ts)] = line

for t in d:
    print t, d[t]
