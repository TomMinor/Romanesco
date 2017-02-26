#include "keyframecomponent.h"

std::vector<KeyframeComponent*> KeyframeComponent::m_keyframedItems;

void KeyframeComponent::registerKeyframedItem(KeyframeComponent* _k)
{
    m_keyframedItems.emplace_back(_k);
}
