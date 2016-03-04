#ifndef BASEDISTANCEFIELDOPERATOR_H
#define BASEDISTANCEFIELDOPERATOR_H

#include <string>
#include <map>
#include <vector>
#include <set>

typedef std::map<std::string, std::string> OperatorParams;

enum class ReturnType
{
    Void,
    Float,
    Int,
    Vec3,
    Mat4
};

static const std::string ReturnLookup[] = {
    std::string("void"),
    std::string("float"),
    std::string("int"),
    std::string("vec3"),
    std::string("mat4")
};

struct Argument
{
    std::string name;
    ReturnType type;
    std::string defaultValue;
};


class BaseSDFOP
{
public:
    BaseSDFOP();
    ~BaseSDFOP();

    virtual std::string getFunctionName();
    virtual std::string getSource();
    std::string getTypeString();
    virtual Argument getArgument(unsigned int index) = 0;
    virtual unsigned int argumentSize() = 0;

    static std::set<std::string> m_headers;


protected:
    ///
    /// \brief m_displayName
    ///
    std::string m_displayName = "Base Operator";

    ReturnType m_returnType;
};

#endif // BASEDISTANCEFIELDOPERATOR_H
