#/bin/bash

GLOBALARGS="-b --timeout 50"
OUTPUTFILENAME=fractal.1%03d.exr
OUTPUTPATH=/transfer/fractals
mkdir -p $OUTPUTPATH

WIDTH=1920
HEIGHT=1080

# spc_sh_070 f115 start:2294 @todo fly towards camera
#SHOTFOLDER=$OUTPUTPATH/spc_sh_070; mkdir -p $SHOTFOLDER;
#./romanesco $GLOBALARGS -s 0 -e 115 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/spc_sh_070.cu --timeout 20

# fra_sh_010 f152 @todo Fix the mandelbulb
# ./romanesco -s 0 -e 10 -f /transfer/fractals/out_%04d.exr -a 1920 -b 1080 -i ./scenes/fra_sh_010.cu --timeout 20

# fra_sh_020 f106 start:2561
#SHOTFOLDER=$OUTPUTPATH/fra_sh_020; mkdir -p $SHOTFOLDER;
#echo "./romanesco $GLOBALARGS -s 0 -e 106 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_020.cu"
#./romanesco $GLOBALARGS -s 0 -e 106 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_020.cu

# fra_sh_030 f90 start:2667
SHOTFOLDER=$OUTPUTPATH/fra_sh_030; mkdir -p $SHOTFOLDER;
echo "./romanesco $GLOBALARGS -s 107 -e 197 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_030.cu"
./romanesco $GLOBALARGS -s 107 -e 197 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_030.cu

# fra_sh_040 f69 start:2757
SHOTFOLDER=$OUTPUTPATH/fra_sh_040; mkdir -p $SHOTFOLDER;
echo "./romanesco $GLOBALARGS -s 198 -e 267 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_040.cu"
./romanesco $GLOBALARGS -s 198 -e 267 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_040.cu

# fra_sh_050 f78 start:2826
SHOTFOLDER=$OUTPUTPATH/fra_sh_050; mkdir -p $SHOTFOLDER;
echo "./romanesco $GLOBALARGS -s 268 -e 346 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_050.cu"
./romanesco $GLOBALARGS -s 268 -e 346 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_050.cu

# fra_sh_060 f92 start:2904
SHOTFOLDER=$OUTPUTPATH/fra_sh_060; mkdir -p $SHOTFOLDER;
echo "./romanesco $GLOBALARGS -s 347 -e 439 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_060.cu"
./romanesco $GLOBALARGS -s 347 -e 439 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_060.cu

# fra_sh_070 f61 start:2996
SHOTFOLDER=$OUTPUTPATH/fra_sh_070; mkdir -p $SHOTFOLDER;
echo "./romanesco $GLOBALARGS -s 440 -e 501 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_070.cu"
./romanesco $GLOBALARGS -s 440 -e 501 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_070.cu
