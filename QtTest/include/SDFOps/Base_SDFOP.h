#ifndef BASEDISTANCEFIELDOPERATOR_H
#define BASEDISTANCEFIELDOPERATOR_H

#include <string>
#include <map>
#include <vector>

typedef std::map<std::string, std::string> OperatorParams;

class BaseSDFOP
{
public:
    BaseSDFOP();
    ~BaseSDFOP();

    virtual std::string getSource();

protected:
    ///
    /// \brief m_displayName
    ///
    std::string m_displayName = "Base Operator";

    std::vector<std::string> m_headers;

    OperatorParams m_params;
};

#endif // BASEDISTANCEFIELDOPERATOR_H
