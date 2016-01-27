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

    virtual std::string getSource() = 0;

protected:
    ///
    /// \brief m_displayName
    ///
    std::string m_displayName = "Base Operator";

    ///
    /// \brief m_source Inline source code snippet
    ///
    std::string m_source = "";

    std::vector<std::string> m_headers;

    OperatorParams m_params;
};

#endif // BASEDISTANCEFIELDOPERATOR_H
