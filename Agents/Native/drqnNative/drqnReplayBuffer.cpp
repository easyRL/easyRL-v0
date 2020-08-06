#include "drqnReplayBuffer.hpp"
#include <time.h>
#include <random>
#include <cstring>

ReplayBuffer::ReplayBuffer(int inStateSize, int inBufferSize, int inBatchSize, int inHistorySize)
{
  curSize = 0;
  ind = 0;
  stateSize = inStateSize;
  bufferSize = inBufferSize;
  batchSize = inBatchSize;
  historySize = inHistorySize;

  srand(time(nullptr));
  states = new float[bufferSize * stateSize];
  actions = new int64_t[bufferSize];
  rewards = new float[bufferSize];
  dones = new int64_t[bufferSize];
}

void ReplayBuffer::add(float* state, int64_t action, float reward, int64_t done)
{
  memcpy(&states[ind * stateSize], state, sizeof(float)*stateSize);
  actions[ind] = action;
  rewards[ind] = reward;
  dones[ind] = done;
  ind = (ind+1)%bufferSize;
  if (curSize < bufferSize)
  {
    curSize++;
  }
}

void ReplayBuffer::sample(float* bStates, int64_t* bActions, float* bRewards, float* bNextStates, int64_t* bDones)
{
  for (int i=0; i<batchSize; i++)
  {
    int sInd = rand()%curSize;
    int nextSind = (sInd+1)%bufferSize;
    
    memcpy(&bStates[stateSize*(historySize*(i+1)-1)], &states[sInd * stateSize], sizeof(float)*stateSize);
    int curSInd = sInd;
    int j=1;
    for (; (curSInd != 0 || curSize == bufferSize) && curSInd != ind && j<historySize; j++)
    {
      curSInd = (curSInd-1+curSize)%curSize;
      if (dones[curSInd])
      {
        break;
      }
      memcpy(&bStates[stateSize*((i+1)*historySize-1-j)], &states[curSInd * stateSize], sizeof(float)*stateSize);
    }
    for (; j<historySize; j++)
    {
      memset(&bStates[stateSize*((i+1)*historySize-1-j)], 0, sizeof(float)*stateSize);
    }
    
    bActions[i] = actions[sInd];
    bRewards[i] = rewards[sInd];
    bDones[i] = dones[sInd];

    if (dones[sInd] || nextSind == ind)
    {
      memset(&bNextStates[i*historySize*stateSize], 0, sizeof(float)*historySize*stateSize);
    }
    else
    {
      memcpy(&bNextStates[i*historySize*stateSize], &bStates[stateSize*(i*historySize + 1)], sizeof(float)*(historySize-1)*stateSize);
      memcpy(&bNextStates[stateSize*((i+1)*historySize - 1)], &states[nextSind*stateSize], sizeof(float)*stateSize);
    }
  }
}

void ReplayBuffer::recent(float* bStates, float* curState)
{
  int sInd = (ind-1+curSize)%curSize;
  
  int curSInd = sInd;
  int j=0;
  for (; (curSInd != 0 || curSize == bufferSize) && curSInd != ind && j<historySize-1; j++)
  {
    curSInd = (curSInd-1+curSize)%curSize;
    if (dones[curSInd])
    {
      break;
    }
    memcpy(&bStates[(historySize-2-j)*stateSize], &states[curSInd*stateSize], sizeof(float)*stateSize);
  }
  for (; j<historySize-1; j++)
  {
    memset(&bStates[(historySize-2-j)*stateSize], 0, sizeof(float)*stateSize);
  }
  
  memcpy(&bStates[(historySize-1)*stateSize], curState, sizeof(float)*stateSize);
}

ReplayBuffer::~ReplayBuffer()
{
  delete [] states;
  delete [] actions;
  delete [] rewards;
  delete [] dones;
}
