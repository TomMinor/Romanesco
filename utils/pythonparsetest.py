#!/usr/bin/python

import sys

cuFile = sys.argv[1]
hook = sys.argv[2]
ptxFile = sys.argv[3]

def fileToStringList(file):
	with open(file, 'r') as f:
		return f.read()

def findFunction(lines, funcname):
	startLine = -1
	endLine = -1

	extern = False

	for i, line in enumerate(lines):
		if ".func" in line and funcname in line:
			if ".extern" in line:
				extern  = True
			startLine = i
			break

	for i, line in enumerate( lines[startLine:] ):
		if extern:
			if ";" in line:
				endLine = i + 1
				break
		else:
			if "}" in line:
				endLine = i + 1
				break

	return startLine, endLine + startLine

# lines = []
# with open("/home/tom/hook.cu.ptx", "r") as f:
# 	lines = f.readlines()

# start, end = findFunction(lines, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags")

lines = []
with open("/home/tom/src/Romanesco/Romanesco_Src/ptx/raymarchtest.cu.ptx", "r") as f:
	lines = f.readlines()
start, end = findFunction(lines, "distancehit_hook")

print start, end

for line in lines[start:end]:
 	print line,

def patchptx(file, hook, ptxfile):
	hookString = ".visible .func  (.param .b32 func_retval0){0}".format(hook)

	ptx = []
	ptxHook = fileToStringList(ptxfile).split('\n')
	end = ptxHook.index(".address_size 64")
	ptxHook = "\n".join( ptxHook[end + 1:] )

	with open(file, 'r') as f:
		hookExtern = ".extern .func  (.param .b32 func_retval0)"
		foundExtern = False
		for line in f:
			if foundExtern:
				if ";" in line:
					foundExtern = False
					ptx.append(ptxHook)	
			elif hookExtern in line:
				foundExtern = True
			else:
				ptx.append( line )

	print "".join(ptx)

	with open("/home/tom/src/Romanesco/Romanesco_Src/ptx/raymarchtest.cu.ptx", 'w') as f:
		f.write( "".join(ptx) )

	

# patchptx(cuFile, hook, ptxFile)