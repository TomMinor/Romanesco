#!/usr/bin/python

import sys

cuFile = sys.argv[1]
hook = sys.argv[2]
ptxFile = sys.argv[3]

def fileToStringList(file):
	with open(file, 'r') as f:
		return f.read()

def findFunction(filename, funcname):
	startLine = -1
	endLine = -1

	with open(filename, "r") as f:
		for i, line in enumerate(f):
			if ".func" in line and funcname in line:
				startLine = i

findFunction("/home/tom/hook.cu.ptx", "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags")

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