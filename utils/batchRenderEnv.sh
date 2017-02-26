
#/bin/bash


OVERRIDEFRAMES=true

if [[ -z ${START+x} && -z ${END+x} ]];
then 
	OVERRIDEFRAMES=false
fi


GLOBALARGS="-b $EXTRAARGS"
PREVIEWARGS="--samples 1 --timeout 1"
FINALARGS="--samples 2 --timeout 1"

OUTPUTFILENAME=fractal.1%03d.exr
OUTPUTPATH=$KYRANPC
mkdir -p $OUTPUTPATH

EXECUTABLE=./romanesco
WIDTH=1024
HEIGHT=1024

TILEX=16
TILEY=16


declare -A ENVTEXTURES
# ================== Core Environment Map Shots ======================
ENVTEXTURES[fra_sh_030]="0 	90 0" #2667  90
ENVTEXTURES[fra_sh_040]="198	267 0" #2757  69
ENVTEXTURES[fra_sh_050]="0 	78 0" #2826  78


ENVTEXTURES[fra_sh_030_a]="120 	210 0 fra_sh_030" #2667  90
ENVTEXTURES[fra_sh_030_b]="120 	210 0 fra_sh_030" #2667  90

ENVTEXTURES[fra_sh_040_a]="198	267 0 fra_sh_040" #2757  69
ENVTEXTURES[fra_sh_040_b]="198	267 0 fra_sh_040" #2757  69

ENVTEXTURES[fra_sh_050_a]="268 	346 0 fra_sh_050" #2826  78
ENVTEXTURES[fra_sh_050_b]="268 	346 0 fra_sh_050" #2826  78


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
		$EXECUTABLE --environmentCamera $GLOBALARGS $FINALARGS -s $startframe -e $endframe --tileX $TILEX --tileY $TILEY--offset $frameoffset -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/${shotname}_env.cu
	else
		SHOTFOLDER=$OUTPUTPATH/$shotname/images/sphericalFractals; mkdir -p $SHOTFOLDER;

		printf "Starting environment map render for shot $(tput setaf $highlight_color)$shotname$(tput sgr0)... Start: $(tput setaf $frame_color)[%s]$(tput sgr0)\tEnd: $(tput setaf $frame_color)[%s]$(tput sgr0)\tOffset: $(tput setaf $frame_color)[%s]$(tput sgr0)\n" $startframe $endframe $frameoffset
		printf "Output directory: $(tput setaf $frame_color)%s$(tput sgr0)" $SHOTFOLDER
		$EXECUTABLE --environmentCamera $GLOBALARGS $PREVIEWARGS -s $startframe -e $endframe --tileX $TILEX --tileY $TILEY --offset $frameoffset -f $SHOTFOLDER/$OUTPUTFILENAME --width $WIDTH --height $HEIGHT -i ./scenes/${shotname}_env.cu
	fi
	echo
done
