#ifndef INPUTHANDLER_H
#define INPUTHANDLER_H

#include <unordered_map>



#include "command.h"

class InputHandler
{
public:
    InputHandler();

    void handleInput();
private:
    std::unordered_map<std::string, Command*> m_commands;
};

#endif // INPUTHANDLER_H
