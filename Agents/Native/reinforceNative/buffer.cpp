#include "buffer.h"
#include <string.h>

Buffer::Buffer(int inStateSize)
{
  stateSize = inStateSize;
}

void Buffer::add(float* state, int64_t action, float reward)
{
  for (int i=0; i<stateSize; i++)
  {
    float val = state[i];
    states.push_back(val);
  }
  actions.push_back(action);
  rewards.push_back(reward);
}
  
void Buffer::processDiscountedRewards(float gamma)
{
  if (rewards.size() != 0)
  {
    int base = rewardsToGo.size();
    rewardsToGo.resize(rewardsToGo.size() + rewards.size());
    rewardsToGo[rewardsToGo.size()-1] = rewards.back();
    for (int i=1; i<rewards.size(); i++)
    {
      int ind = rewardsToGo.size()-1-i;
      rewardsToGo[ind] = rewards[rewards.size()-1-i] + gamma * rewardsToGo[ind+1];
    }
    
    rewards.clear();
  }
}

bool Buffer::getPrevState(float* prevState)
{
  if (states.size() == 0)
  {
    return false;
  }
  
  memcpy(prevState, &states[states.size()-stateSize], sizeof(float)*stateSize);
  return true;
}
    
bool Buffer::getPrevAction(int64_t& prevAction)
{
  if (states.size() == 0)
  {
    return false;
  }
  
  prevAction = actions.back();
  return true;
}

float Buffer::getLastReward()
{
  return rewards[rewards.size() - 1];
}

int Buffer::storedSize()
{
  return actions.size();
}

int Buffer::processedSize()
{
  return rewardsToGo.size();
}

int Buffer::unProcessedSize()
{
  return storedSize() - processedSize();
}

void Buffer::clearProcessed()
{
  int numUnprocessed = rewards.size();

  if (numUnprocessed == 0)
  {
    states.clear();
    actions.clear();
    rewardsToGo.clear();
  }
  else
  {
    memcpy(&states[0], &states[states.size() - stateSize*numUnprocessed], sizeof(float)*stateSize*numUnprocessed);
    states.resize(stateSize*numUnprocessed);
  
    memcpy(&actions[0], &actions[actions.size() - numUnprocessed], sizeof(int64_t)*numUnprocessed);
    actions.resize(numUnprocessed);
    
    rewardsToGo.clear();
  }
}
