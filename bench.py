#!/usr/bin/python

import re
import commands

def imp(x, n=7):
    for red in x:
        print '\t'.join(map(lambda s: str(s)[:n], red))

vremena = []
greska = []
dimenzije = range(4,21)
a = ()
for i in dimenzije:
    if i < 11:
        broj = 2**i
    else:
        broj = 1024 * (i-9)
    cmd = 'GPUCHOL ' + str(broj) +" " +str(broj)+'s.h5'
    t = 0
    for j in range(1, 11):
      x = commands.getstatusoutput(cmd)
      g = re.search('UKUPNO:\s([\d.]*)', x[1])
      t += float(g.group(1))
      
    #vremena.append(t/10)
    g = re.search('GPU\s([\d.-]+)', x[1])
    #greska.append(float(g.group(1)))
    
    vremena.append((broj, t/10, float(g.group(1))))
    
    #print
    #print broj
imp(vremena, n=15)
    #print greska[-1]
