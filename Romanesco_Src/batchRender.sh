#/bin/bash


OVERRIDEFRAMES=true

if [[ -z ${START+x} && -z ${END+x} ]];
then 
	OVERRIDEFRAMES=false
fi


GLOBALARGS="-b $EXTRAARGS"
PREVIEWARGS="--samples 1 --timeout 10"
FINALARGS="--samples 2 --timeout 1"

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
# ================== Core Shots ======================
SHOTS[spc_sh_070]="0 	115 0" #2294  115
SHOTS[fra_sh_010]="0 	152 0" #2409  152
SHOTS[fra_sh_020]="0 	106 0" #2561  106
SHOTS[fra_sh_030]="120 	210 0" #2667  90
SHOTS[fra_sh_040]="198	267 0" #2757  69
SHOTS[fra_sh_050]="268 	346 0" #2826  78
SHOTS[fra_sh_060]="347 	439 0" #2904  92
SHOTS[fra_sh_070]="440 	501 0" #2996  61

# ================== Fix Up Shots ======================
SHOTS[fra_sh_010_a]="0 	30 0 fra_sh_010"
SHOTS[fra_sh_010_b]="30 60 30 fra_sh_010"
SHOTS[fra_sh_010_c]="60 90 60 fra_sh_010"
SHOTS[fra_sh_010_d]="90 120 90 fra_sh_010"
SHOTS[fra_sh_010_e]="120 152 120 fra_sh_010"

SHOTS[fra_sh_020_a]="0 0 44 fra_sh_020"

SHOTS[fra_sh_030_a]="0 0 56 fra_sh_030"

SHOTS[fra_sh_040_a]="13 28 -1 fra_sh_040"
SHOTS[fra_sh_040_b]="0 0 36 fra_sh_040"

SHOTS[fra_sh_050_a]="0 0 9 fra_sh_050"
SHOTS[fra_sh_050_b]="0 0 18 fra_sh_050"

SHOTS[fra_sh_060_a]="0 0 70 fra_sh_060"
SHOTS[fra_sh_060_b]="0 0 76 fra_sh_060"

SHOTS[fra_sh_070_a]="0 0 32 fra_sh_070"
SHOTS[fra_sh_070_b]="0 0 41 fra_sh_070"

# Shots  broken up into pieces
SHOTS[spc_sh_070_a]="0 	 28 0 spc_sh_070" # 152
SHOTS[spc_sh_070_b]="28  58 28 spc_sh_070"
SHOTS[spc_sh_070_c]="58  86 58 spc_sh_070"
SHOTS[spc_sh_070_d]="86 115 86 spc_sh_070"

# Extra frames for wiggle room
SHOTS[spc_sh_070_e]="116 130 116 spc_sh_070"
SHOTS[spc_sh_070_f]="131 150 130 spc_sh_070"
SHOTS[spc_sh_070_g]="150 180 150 spc_sh_070"



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

	shotname=$shot
	parentshotname=${framedata[3]}

	# Is the parent shot defined?
	if [[ -n  "$parentshotname" ]]
	then
		shotname="$parentshotname"
	fi

	startframe="${framedata[0]}"
	endframe="${framedata[1]}"
	frameoffset="${framedata[2]}"

	if [ "$OVERRIDEFRAMES" = true ]; then
		echo "$(tput setaf $error_color)Overriding frame range to $START:$END$(tput sgr0)"
		endframe=$END
		startframe=$START
        frameoffset=$START
	fi

	if [[ -z ${FINAL+x} ]];
	then
		SHOTFOLDER=$OUTPUTPATH/$shotname/images/fractals_hd; mkdir -p $SHOTFOLDER;

		printf "Starting final quality render for shot $(tput setaf $highlight_color)$shotname$(tput sgr0)... Start: $(tput setaf $frame_color)[%s]$(tput sgr0)\tEnd: $(tput setaf $frame_color)[%s]$(tput sgr0)\tOffset: $(tput setaf $frame_color)[%s]$(tput sgr0)\n" $startframe $endframe $frameoffset
		printf "Output directory: $(tput setaf $frame_color)%s$(tput sgr0)" $SHOTFOLDER
		$EXECUTABLE $GLOBALARGS $FINALARGS -s $startframe -e $endframe --tileX $TILEX --tileY $TILEY--offset $frameoffset -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/$shotname.cu
	else
		SHOTFOLDER=$OUTPUTPATH/$shotname/images/fractals; mkdir -p $SHOTFOLDER;

		printf "Starting render for shot $(tput setaf $highlight_color)$shotname$(tput sgr0)... Start: $(tput setaf $frame_color)[%s]$(tput sgr0)\tEnd: $(tput setaf $frame_color)[%s]$(tput sgr0)\tOffset: $(tput setaf $frame_color)[%s]$(tput sgr0)\n" $startframe $endframe $frameoffset
		printf "Output directory: $(tput setaf $frame_color)%s$(tput sgr0)" $SHOTFOLDER
		$EXECUTABLE $GLOBALARGS $PREVIEWARGS -s $startframe -e $endframe --tileX $TILEX --tileY $TILEY --offset $frameoffset -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/$shotname.cu
	fi
        echo
done


echo ".. Removing duplicate environment shot inputs..."
sorted_unique_ids=($(echo "${@}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

printf ".. Environment Shots to process: $(tput setaf $highlight_color) ${sorted_unique_ids[*]} $(tput sgr0)";
tput sgr0
echo

echo "-------------------------------------------------"
echo ".. Removing invalid environment shot inputs..."
for shot in "${sorted_unique_ids[@]}"
do
	if array_contains "$shot" "${!ENVTEXTURES[@]}"; then
		: # Shot is valid
	else
		delete=($shot)
		sorted_unique_ids=("${sorted_unique_ids[@]/$delete}")

		echo "$(tput setaf $error_color) Warning: Environment shot $shot doesn't exist, skipping $(tput sgr0)" 
	fi
done
echo "-------------------------------------------------"
echo ".. Final list of environment shots to process..."
printf ".. Shots to process: "; printf "$(tput setaf $highlight_color) ${sorted_unique_ids[*]} $(tput sgr0)\n";
echo

final_shots=(${sorted_unique_ids[@]})
for shot in "${final_shots[@]}"
do
	framedata=(${ENVTEXTURES[$shot]})

	shotname=$shot
	parentshotname=${framedata[3]}

	# Is the parent shot defined?
	if [[ -n  "$parentshotname" ]]
	then
		shotname="$parentshotname"
	fi

	startframe="${framedata[0]}"
	endframe="${framedata[1]}"
	frameoffset="${framedata[2]}"

	if [ "$OVERRIDEFRAMES" = true ]; then
		echo "$(tput setaf $error_color)Overriding frame range to $START:$END$(tput sgr0)"
		endframe=$END
		startframe=$START
        frameoffset=$START
	fi

	if [[ -z ${FINAL+x} ]];
	then
		SHOTFOLDER=$OUTPUTPATH/$shotname/images/sphericalFractals_hd; mkdir -p $SHOTFOLDER;

		printf "Starting final quality environment map render for shot $(tput setaf $highlight_color)$shotname$(tput sgr0)... Start: $(tput setaf $frame_color)[%s]$(tput sgr0)\tEnd: $(tput setaf $frame_color)[%s]$(tput sgr0)\tOffset: $(tput setaf $frame_color)[%s]$(tput sgr0)\n" $startframe $endframe $frameoffset
		printf "Output directory: $(tput setaf $frame_color)%s$(tput sgr0)" $SHOTFOLDER
		$EXECUTABLE $GLOBALARGS $FINALARGS --environmentCamera -s $startframe -e $endframe --tileX $TILEX --tileY $TILEY--offset $frameoffset -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/$shotname.cu
	else
		SHOTFOLDER=$OUTPUTPATH/$shotname/images/sphericalFractals; mkdir -p $SHOTFOLDER;

		printf "Starting environment map render for shot $(tput setaf $highlight_color)$shotname$(tput sgr0)... Start: $(tput setaf $frame_color)[%s]$(tput sgr0)\tEnd: $(tput setaf $frame_color)[%s]$(tput sgr0)\tOffset: $(tput setaf $frame_color)[%s]$(tput sgr0)\n" $startframe $endframe $frameoffset
		printf "Output directory: $(tput setaf $frame_color)%s$(tput sgr0)" $SHOTFOLDER
		$EXECUTABLE $GLOBALARGS $PREVIEWARGS --environmentCamera -s $startframe -e $endframe --tileX $TILEX --tileY $TILEY --offset $frameoffset -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/$shotname.cu
	fi
	echo
done
