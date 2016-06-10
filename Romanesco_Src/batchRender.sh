#/bin/bash


OVERRIDEFRAMES=true

if [[ -z ${START+x} && -z ${END+x} ]];
then 
	OVERRIDEFRAMES=false
fi


GLOBALARGS="-b --timeout 20 $EXTRAARGS"
OUTPUTFILENAME=fractal.1%03d.exr
OUTPUTPATH=$KYRANPC
mkdir -p $OUTPUTPATH

EXECUTABLE=./romanesco
WIDTH=1920
HEIGHT=1080

TILEX=16
TILEY=16

# Associative array for easy shot data editing
declare -A SHOTS
SHOTS[spc_sh_070]="0 	115 0" #2294  115
SHOTS[fra_sh_010]="0 	152 0" #2409  152
SHOTS[fra_sh_020]="0 	106 0" #2561  106
SHOTS[fra_sh_030]="120 	210 0" #2667  90
SHOTS[fra_sh_040]="198	267 0" #2757  69
SHOTS[fra_sh_050]="268 	346 0" #2826  78
SHOTS[fra_sh_060]="347 	439 0" #2904  92
SHOTS[fra_sh_070]="440 	501 0" #2996  61

# Bad frames
# 66 - 120 # Gaps
# ~149 - 160 # Boring shape

# Good bits
# 20   # Very organic
# 120
# 160 - 240 # Interesting shape, cauliflower
# 450
# 650

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
frame_color=4 # Blue
error_color=1 # Red

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~~~ Romanesco Render (Batch) ~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Global Args: \"$GLOBALARGS\""
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

	if [ "$OVERRIDEFRAMES" = true ]; then
		echo "$(tput setaf $error_color)Overriding frame range to $START:$END$(tput sgr0)"
		endframe=$END
		startframe=$START
	fi

	SHOTFOLDER=$OUTPUTPATH/$shot/images/fractals; mkdir -p $SHOTFOLDER;
	$EXECUTABLE $GLOBALARGS -s $startframe -e $endframe --tileX $TILEX --tileY $TILEY --offset $frameoffset -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/$shot.cu
done


for shot in "${final_shots[@]}"
do
	framedata=(${SHOTS[$shot]})
	startframe="${framedata[0]}"
	endframe="${framedata[1]}"
	frameoffset="${framedata[2]}"

	if [ "$OVERRIDEFRAMES" = true ]; then
		:
		# $endframe = $END
		# $startframe = $START
	fi

	printf "Starting final quality render for shot $(tput setaf $highlight_color)$shot$(tput sgr0)... Start: $(tput setaf $frame_color)[%s]$(tput sgr0)\tEnd: $(tput setaf $frame_color)[%s]$(tput sgr0)\tOffset: $(tput setaf $frame_color)[%s]$(tput sgr0)\n" $startframe $endframe $frameoffset

	SHOTFOLDER=$OUTPUTPATH/$shot/images/fractals; mkdir -p $SHOTFOLDER;
	$EXECUTABLE $GLOBALARGS --samples 3 -s $startframe -e $endframe --tileX $TILEX --tileY $TILEY--offset $frameoffset -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/$shot.cu
done
