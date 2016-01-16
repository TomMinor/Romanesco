#ifndef BASEDISTANCEFIELDOPERATOR_H
#define BASEDISTANCEFIELDOPERATOR_H

#include <string>
#include <map>

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
    /// \brief m_sourcePath Source code snippet filepath
    ///
    std::string m_sourcePath = "";

    ///
    /// \brief m_source Inline source code snippet
    ///
    std::string m_source = "";

    OperatorParams m_params;
};

#endif // BASEDISTANCEFIELDOPERATOR_H
