#!/usr/bin/python

import sys

if len(sys.argv) != 2:
	print "Invalid arguments, expected bib file"
	exit(1)

def wordCount(string):
	return len(''.join(c if c.isalnum() else ' ' for c in string).split())

bibfile = sys.argv[1]

totalWordCount = 0

with open(bibfile) as f:
	for line in f.readlines():
		splitLine = line.split()
		if len(splitLine) > 0 and splitLine[0] == 'annote':
			bodyWords = ' '.join(splitLine[2:])[2:-2]
			totalWordCount += wordCount(bodyWords)

print totalWordCount