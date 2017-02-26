#!/usr/bin/fish

# Generate file structure for an SDF op

set include_dir "./include/SDFOps"
set src_dir "./src/SDFOps"

set base_header "#ifndef NAME_SDFOP_H
#define NAME_SDFOP_H

#include \"Base_SDFOP.h\"

class Name_SDFOP : public BaseSDFOP
{
public:
    Name_SDFOP();
    ~Name_SDFOP();
};

#endif // NAME_SDFOP_H
"

set base_src "#include \"Name_SDFOP.h\"

Name_SDFOP::Name_SDFOP()
{
}

Name_SDFOP::~Name_SDFOP()
{
}
"

function capitalize
    set input "$argv"
    echo "$input" | tr '[A-Z]' '[a-z]' | sed 's/\(^\| \)\([a-z]\)/\1\u\2/g'
end

function uppercase
    set input "$argv"
    echo "$input" | tr '[:lower:]' '[:upper:]'
end

if count $argv > /dev/null
	set uppername (uppercase $argv[1])
	set capitalname (capitalize $argv[1])

	set generated_header \n( echo $base_header | sed "s/NAME/$uppername/g; s/Name/$capitalname/g;  ")
	set header_file "$include_dir/"$capitalname"_SDFOP.h"
	if not test -e $header_file
		echo $generated_header > $header_file
	end

	set generated_src \n( echo $base_src | sed "s/NAME/$uppername/g; s/Name/$capitalname/g;  ")

	set src_file "$src_dir/"$capitalname"_SDFOP.cpp"
	if not test -e $src_file
		echo $generated_src > $src_file
	end
end