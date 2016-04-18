#ifndef STRINGUTILITIES_H
#define STRINGUTILITIES_H

#include <vector>
#include <string>

/**
 * @brief FileToVector
 * @param _filename
 * @return
 */
std::vector<std::string> FileToVector(const std::string& _filename);

/**
 * @brief StringToVector
 * @param _str
 * @return
 */
std::vector<std::string> StringToVector(const std::string& _str);

/**
 * @brief findString
 * @param _lines
 * @param _searchString
 * @param o_result
 * @return
 */
bool findString(std::vector<std::string>& _lines,
                const std::string& _searchString,
                std::vector<std::string>::iterator* o_result);


#endif // STRINGUTILITIES_H
