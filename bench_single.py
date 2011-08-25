#!/usr/bin/python

import re
import commands

def imp(x, n=7):
    for red in x:
        print '\t'.join(map(lambda s: str(s)[:n], red))

vremena = []
greska = []
dimenzije = range(4,22)
a = ()
for i in dimenzije:
    print i
    if i < 11:
        broj = 2**i
    else:
        broj = 1024 * (i-9)
    cmd = 'GPUCHOL_SINGLE ' + str(broj) +" " +str(broj)+'s.h5'
    t = 0
    for j in range(1, 11):
      x = commands.getstatusoutput(cmd)
      g = re.search('UKUPNO:\s([\d.]*)', x[1])
      t += float(g.group(1))

    g = re.search('GPU\s([\d.-]+)', x[1])

    vremena.append((broj, t/10, float(g.group(1))))

imp(vremena, n=15)
