#ifndef BASEDISTANCEFIELDOPERATOR_H
#define BASEDISTANCEFIELDOPERATOR_H

#include <string>
#include <map>
#include <vector>
#include <set>

typedef std::map<std::string, std::string> OperatorParams;

class BaseSDFOP
{
public:
    BaseSDFOP();
    ~BaseSDFOP();

    virtual std::string getSource();

    virtual std::string getDefaultArg(unsigned int index);

    static std::set<std::string> m_headers;

protected:
    ///
    /// \brief m_displayName
    ///
    std::string m_displayName = "Base Operator";



    OperatorParams m_params;
};

#endif // BASEDISTANCEFIELDOPERATOR_H
