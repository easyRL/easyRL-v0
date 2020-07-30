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
  float (*bStatesAccess)[historySize][stateSize] = (float (*)[historySize][stateSize])bStates;
  float (*bNextStatesAccess)[historySize][stateSize] = (float (*)[historySize][stateSize])bNextStates;
  float (*statesAccess)[stateSize] = (float (*)[stateSize])states;

  for (int i=0; i<batchSize; i++)
  {
    int sInd = rand()%curSize;
    int nextSind = (sInd+1)%bufferSize;
    
    memcpy(&bStatesAccess[i][historySize-1][0], &statesAccess[sInd][0], sizeof(float)*stateSize);
    int curSInd = sInd;
    int j=1;
    for (; (curSInd != 0 || curSize == bufferSize) && curSInd != ind && j<historySize; j++)
    {
      curSInd = (curSInd-1+curSize)%curSize;
      if (dones[curSInd])
      {
        break;
      }
      memcpy(&bStatesAccess[i][historySize-1-j][0], &statesAccess[curSInd][0], sizeof(float)*stateSize);
    }
    for (; j<historySize; j++)
    {
      memset(&bStatesAccess[i][historySize-1-j][0], 0, sizeof(float)*stateSize);
    }
    
    bActions[i] = actions[sInd];
    bRewards[i] = rewards[sInd];
    bDones[i] = dones[sInd];
    
    if (dones[sInd] || nextSind == ind)
    {
      memset(&bNextStatesAccess[i][0][0], 0, sizeof(float)*historySize*stateSize);
    }
    else
    {
      memcpy(&bNextStatesAccess[i][0][0], &bStatesAccess[i][1][0], sizeof(float)*(historySize-1)*stateSize);
      memcpy(&bNextStatesAccess[i][historySize-1][0], &statesAccess[nextSind][0], sizeof(float)*stateSize);
    }
  }
}

void ReplayBuffer::recent(float* bStates, float* curState)
{
  float (*bStatesAccess)[stateSize] = (float (*)[stateSize])bStates;
  float (*statesAccess)[stateSize] = (float (*)[stateSize])states;
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
    memcpy(&bStatesAccess[historySize-2-j][0], &statesAccess[curSInd][0], sizeof(float)*stateSize);
  }
  for (; j<historySize-1; j++)
  {
    memset(&bStatesAccess[historySize-2-j][0], 0, sizeof(float)*stateSize);
  }
  
  memcpy(&bStatesAccess[historySize-1][0], curState, sizeof(float)*stateSize);
}

ReplayBuffer::~ReplayBuffer()
{
  delete [] states;
  delete [] actions;
  delete [] rewards;
  delete [] dones;
}
