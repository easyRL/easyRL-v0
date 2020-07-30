#include "dqnReplayBuffer.hpp"
#include <time.h>
#include <random>
#include <cstring>

ReplayBuffer::ReplayBuffer(int inStateSize, int inBufferSize, int inBatchSize)
{
  curSize = 0;
  ind = 0;
  stateSize = inStateSize;
  bufferSize = inBufferSize;
  batchSize = inBatchSize;

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
    memcpy(&bStates[i*stateSize], &states[sInd*stateSize], sizeof(float)*stateSize);
    bActions[i] = actions[sInd];
    bRewards[i] = rewards[sInd];
    bDones[i] = dones[sInd];
    if (bDones[i] || nextSind == ind)
    {
      memset(&bNextStates[i*stateSize], 0, sizeof(float)*stateSize);
    }
    else
    {
      memcpy(&bNextStates[i*stateSize], &states[nextSind*stateSize], sizeof(float)*stateSize);
    }
  }
}

ReplayBuffer::~ReplayBuffer()
{
  delete [] states;
  delete [] actions;
  delete [] rewards;
  delete [] dones;
}
