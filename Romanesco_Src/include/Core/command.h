#ifndef COMMAND_H
#define COMMAND_H

///
/// \brief The Command class
/// Modified from :-
/// [Accessed 2016] http://gameprogrammingpatterns.com/command.html
///
class Command
{
public:
    Command();
    virtual ~Command() {}
    virtual void execute() = 0;
};

#endif // COMMAND_H
