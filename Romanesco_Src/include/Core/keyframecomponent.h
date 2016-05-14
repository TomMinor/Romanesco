#ifndef KEYFRAMECOMPONENT_H
#define KEYFRAMECOMPONENT_H

#include <vector>

///
/// \brief The KeyframeComponent class is a base interface for objects that would like to be keyframable.
///
class KeyframeComponent
{
public:
    KeyframeComponent();

    static void registerKeyframedItem(KeyframeComponent* _k);

private:
    static std::vector<KeyframeComponent*> m_keyframedItems;
};

#endif // KEYFRAMECOMPONENT_H
