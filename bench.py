#!/usr/bin/python

import re
import commands

vremena = []
greska = []
dimenzije = [2**x for x in range(4,10)]
for broj in dimenzije:
	cmd = 'GPUCHOL ' + str(broj) + ' fullspread.h5'
	x = commands.getstatusoutput(cmd)
	g = re.search('UKUPNO:\s([\d.]*)', x[1])
	vremena.append(float(g.group(1)))
	g = re.search('GRESKA\s([\d.-]*)', x[1])
	greska.append(float(g.group(1)))
	print
	print broj
	print vremena[-1]
	print greska[-1]
for broj in range(1024, 10240, 1024):
	cmd = 'GPUCHOL ' + str(broj) + ' fullspread.h5'
	x = commands.getstatusoutput(cmd)
	g = re.search('UKUPNO:\s([\d.]*)', x[1])
	vremena.append(float(g.group(1)))
	g = re.search('GRESKA\s([\d.-]*)', x[1])
	greska.append(float(g.group(1)))
	print
	print broj
	print vremena[-1]
	print greska[-1]
