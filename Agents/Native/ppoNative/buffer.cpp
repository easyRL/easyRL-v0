#include "buffer.h"
#include <string.h>

Buffer::Buffer(int inStateSize)
{
  stateSize = inStateSize;
}

void Buffer::add(float* state, int64_t action, float reward, float valEst, int64_t done)
{
  for (int i=0; i<stateSize; i++)
  {
    float val = state[i];
    states.push_back(val);
  }
  actions.push_back(action);
  rewards.push_back(reward);
  returns.push_back(valEst);
  dones.push_back(done);
}
  
void Buffer::processDiscountedRewards(float gamma, float lambda, bool done)
{
  if (rewards.size() != 0)
  {
    float valEst, lastReward;
    if (done)
    {
      valEst = 0;
    }
    else
    {
      valEst = returns.back();
      lastReward = rewards.back();
      rewards.pop_back();
    }

    int base = rewardsToGo.size();
    rewardsToGo.resize(rewardsToGo.size() + rewards.size());
    advantages.resize(rewardsToGo.size());
    rewardsToGo[rewardsToGo.size()-1] = rewards.back() + gamma * valEst;
    advantages[rewardsToGo.size()-1] = rewardsToGo.back() - returns[rewardsToGo.size()-1];
    for (int i=1; i<rewards.size(); i++)
    {
      int ind = rewardsToGo.size()-1-i;
      rewardsToGo[ind] = rewards[rewards.size()-1-i] + gamma * rewardsToGo[ind+1];
      advantages[ind] = rewards[rewards.size()-1-i] - returns[ind] + gamma * (returns[ind+1] + lambda*advantages[ind+1]);
    }
    
    rewards.clear();
    
    if (!done)
    {
      rewards.push_back(lastReward);
    }
  }
}

bool Buffer::getPrevState(float* prevState)
{
  if (states.size() == 0 || dones.back())
  {
    return false;
  }
  
  memcpy(prevState, &states[states.size()-stateSize], sizeof(float)*stateSize);
  return true;
}
    
bool Buffer::getPrevAction(int64_t& prevAction)
{
  if (states.size() == 0 || dones.back())
  {
    return false;
  }
  
  prevAction = actions.back();
  return true;
}

void Buffer::incrementLastReward(float rewardIncrement)
{
  //assert(rewards.size() > 0);
  rewards[rewards.size() - 1] += rewardIncrement;
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
    advantages.clear();
    returns.clear();
    dones.clear();
  }
  else
  {
    memcpy(&states[0], &states[states.size() - stateSize*numUnprocessed], sizeof(float)*stateSize*numUnprocessed);
    states.resize(stateSize*numUnprocessed);
  
    memcpy(&actions[0], &actions[actions.size() - numUnprocessed], sizeof(int64_t)*numUnprocessed);
    actions.resize(numUnprocessed);
    
    rewardsToGo.clear();
    advantages.clear();
    
    memcpy(&returns[0], &returns[returns.size() - numUnprocessed], sizeof(float)*numUnprocessed);
    returns.resize(numUnprocessed);
    
    memcpy(&dones[0], &dones[dones.size() - numUnprocessed], sizeof(int64_t)*numUnprocessed);
    dones.resize(numUnprocessed);
  }
}
