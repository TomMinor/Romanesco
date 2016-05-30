#/bin/bash

GLOBALARGS="-b"
OUTPUTFILENAME=fractal_1%03d.exr
OUTPUTPATH=/transfer/fractals
mkdir -p $OUTPUTPATH

WIDTH=1920
HEIGHT=1080

# spc_sh_070 f115 @todo fly towards camera
SHOTFOLDER=$OUTPUTPATH/spc_sh_070; mkdir -p $SHOTFOLDER;
./romanesco $GLOBALARGS -s 0 -e 115 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./kernel/spc_sh_070.cu --timeout 20

# fra_sh_010 f152 @todo Fix the mandelbulb
# ./romanesco -s 0 -e 10 -f /transfer/fractals/out_%04d.exr -a 1920 -b 1080 -i ./kernel/fra_sh_010.cu --timeout 20

# fra_sh_020 f106
SHOTFOLDER=$OUTPUTPATH/fra_sh_020; mkdir -p $SHOTFOLDER;
./romanesco $GLOBALARGS -s 0 -e 106 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./kernel/fra_sh_020.cu --timeout 20

# fra_sh_030 f90
SHOTFOLDER=$OUTPUTPATH/fra_sh_030; mkdir -p $SHOTFOLDER;
./romanesco $GLOBALARGS -s 0 -e 115 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./kernel/fra_sh_030.cu --timeout 20

# fra_sh_040 f69
SHOTFOLDER=$OUTPUTPATH/fra_sh_040; mkdir -p $SHOTFOLDER;
./romanesco $GLOBALARGS -s 0 -e 115 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./kernel/fra_sh_040.cu --timeout 20

# fra_sh_050 f78
SHOTFOLDER=$OUTPUTPATH/fra_sh_050; mkdir -p $SHOTFOLDER;
./romanesco $GLOBALARGS -s 0 -e 115 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./kernel/fra_sh_050.cu --timeout 20

# fra_sh_060 f92
SHOTFOLDER=$OUTPUTPATH/fra_sh_060; mkdir -p $SHOTFOLDER;
./romanesco $GLOBALARGS -s 0 -e 115 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./kernel/fra_sh_060.cu --timeout 20

# fra_sh_070 f61
SHOTFOLDER=$OUTPUTPATH/fra_sh_070; mkdir -p $SHOTFOLDER;
./romanesco $GLOBALARGS -s 0 -e 115 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./kernel/fra_sh_070.cu --timeout 20