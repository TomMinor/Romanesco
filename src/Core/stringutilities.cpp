#include "Core/stringutilities.h"

#include <fstream>
#include <sstream>
#include <QDebug>

std::vector<std::string> FileToVector(const std::string& _filename)
{
    std::vector<std::string> result;
    std::ifstream file(_filename);

    if(!file)
    {
        qDebug("Can't open file %s", qPrintable(_filename.c_str()) );
        throw std::runtime_error("Can't open file");
    }

    std::string line;
    while (std::getline(file, line))
    {
        result.push_back(line);
    }

    return result;
}

std::vector<std::string> StringToVector(const std::string& _str)
{
    std::vector<std::string> result;
    {
        std::istringstream src_stream( _str );
        std::string line;
        while(std::getline(src_stream, line))
        {
            result.push_back(line);
        }
    }

    return result;
}

bool findString(std::vector<std::string>& _lines,
                const std::string& _searchString,
                std::vector<std::string>::iterator* o_result)
{
    for(auto line = _lines.begin(); line != _lines.end(); ++line)
    {
        if ( line->find(_searchString) != std::string::npos)
        {
            *o_result = line + 1;
            return true;
        }
    }

    return false;
}

