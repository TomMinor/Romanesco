#/bin/bash

GLOBALARGS="-b --timeout 50 $EXTRAARGS"
OUTPUTFILENAME=fractal.1%03d.exr
OUTPUTPATH=/transfer/fractals
mkdir -p $OUTPUTPATH

EXECUTABLE=./romanesco
WIDTH=1920
HEIGHT=1080

# Associative array for easy shot data editing
declare -A SHOTS
SHOTS[spc_sh_070]="0 	115 0" #2294
SHOTS[fra_sh_010]="0 	152 0" #2409
SHOTS[fra_sh_020]="0 	106 0" #2561
SHOTS[fra_sh_030]="107 	197 0" #2667
SHOTS[fra_sh_040]="198	267 0" #2757
SHOTS[fra_sh_050]="268 	346 0" #2826
SHOTS[fra_sh_060]="347 	439 0" #2904
SHOTS[fra_sh_070]="440 	501 0" #2996

array_contains () {
    local seeking=$1; shift
    local in=1
    for element; do
        if [[ $element == $seeking ]]; then
            in=0
            break
        fi
    done
    return $in
}

# Black        0;30     Dark Gray     1;30
# Red          0;31     Light Red     1;31
# Green        0;32     Light Green   1;32
# Brown/Orange 0;33     Yellow        1;33
# Blue         0;34     Light Blue    1;34
# Purple       0;35     Light Purple  1;35
# Cyan         0;36     Light Cyan    1;36
# Light Gray   0;37     White         1;37

highlight_color=3 # Yellow
frame_color=4 
error_color=1 # Red

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~~~ Romanesco Render (Batch) ~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo
echo "-------------------------------------------------"
echo ".. Removing duplicate shot inputs..."
sorted_unique_ids=($(echo "${@}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

printf ".. Shots to process: $(tput setaf $highlight_color) ${sorted_unique_ids[*]} $(tput sgr0)";
tput sgr0
echo

echo "-------------------------------------------------"
echo ".. Removing invalid shot inputs..."
for shot in "${sorted_unique_ids[@]}"
do
	if array_contains "$shot" "${!SHOTS[@]}"; then
		: # Shot is valid
	else
		delete=($shot)
		sorted_unique_ids=("${sorted_unique_ids[@]/$delete}")

		echo "$(tput setaf $error_color) Warning: Shot $shot doesn't exist, skipping $(tput sgr0)" 
	fi
done
echo "-------------------------------------------------"
echo ".. Final list of shots to process..."
printf ".. Shots to process: "; printf "$(tput setaf $highlight_color) ${sorted_unique_ids[*]} $(tput sgr0)\n";
echo

final_shots=(${sorted_unique_ids[@]})
for shot in "${final_shots[@]}"
do
	framedata=(${SHOTS[$shot]})
	startframe="${framedata[0]}"
	endframe="${framedata[1]}"
	frameoffset="${framedata[2]}"

	printf "Starting render for shot $(tput setaf $highlight_color)$shot$(tput sgr0)... Start: $(tput setaf $frame_color)[%s]$(tput sgr0)\tEnd: $(tput setaf $frame_color)[%s]$(tput sgr0)\tOffset: $(tput setaf $frame_color)[%s]$(tput sgr0)\n" $startframe $endframe $frameoffset

	SHOTFOLDER=$OUTPUTPATH/$shot; mkdir -p $SHOTFOLDER;
	$EXECUTABLE $GLOBALARGS -s $startframe -e $endframe --offset $frameoffset -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/$shot.cu
done

# spc_sh_070 f115 start:2294 @todo fly towards camera
#SHOTFOLDER=$OUTPUTPATH/spc_sh_070; mkdir -p $SHOTFOLDER;
#./romanesco $GLOBALARGS -s 0 -e 115 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/spc_sh_070.cu --timeout 20

# fra_sh_010 f152
# SHOTFOLDER=$OUTPUTPATH/fra_sh_010; mkdir -p $SHOTFOLDER;
# echo "./romanesco $GLOBALARGS -s 0 -e 152 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_010.cu"
# ./romanesco $GLOBALARGS -s 0 -e 152 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_010.cu

# fra_sh_020 f106 start:2561
#SHOTFOLDER=$OUTPUTPATH/fra_sh_020; mkdir -p $SHOTFOLDER;
#echo "./romanesco $GLOBALARGS -s 0 -e 106 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_020.cu"
#./romanesco $GLOBALARGS -s 0 -e 106 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_020.cu

# # fra_sh_030 f90 start:2667
# SHOTFOLDER=$OUTPUTPATH/fra_sh_030; mkdir -p $SHOTFOLDER;
# echo "./romanesco $GLOBALARGS -s 107 -e 197 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_030.cu"
# ./romanesco $GLOBALARGS -s 107 -e 197 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_030.cu

# # fra_sh_040 f69 start:2757
# SHOTFOLDER=$OUTPUTPATH/fra_sh_040; mkdir -p $SHOTFOLDER;
# echo "./romanesco $GLOBALARGS -s 198 -e 267 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_040.cu"
# ./romanesco $GLOBALARGS -s 198 -e 267 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_040.cu

# # fra_sh_050 f78 start:2826
# SHOTFOLDER=$OUTPUTPATH/fra_sh_050; mkdir -p $SHOTFOLDER;
# echo "./romanesco $GLOBALARGS -s 268 -e 346 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_050.cu"
# ./romanesco $GLOBALARGS -s 268 -e 346 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_050.cu

# # fra_sh_060 f92 start:2904
# SHOTFOLDER=$OUTPUTPATH/fra_sh_060; mkdir -p $SHOTFOLDER;
# echo "./romanesco $GLOBALARGS -s 347 -e 439 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_060.cu"
# ./romanesco $GLOBALARGS -s 347 -e 439 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_060.cu

# # fra_sh_070 f61 start:2996
# SHOTFOLDER=$OUTPUTPATH/fra_sh_070; mkdir -p $SHOTFOLDER;
# echo "./romanesco $GLOBALARGS -s 440 -e 501 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_070.cu"
# ./romanesco $GLOBALARGS -s 440 -e 501 -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/fra_sh_070.cu
