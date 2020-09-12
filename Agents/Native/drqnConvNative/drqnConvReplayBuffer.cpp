#include "drqnConvReplayBuffer.hpp"
#include <time.h>
#include <random>
#include <cstring>
#include <iostream>

using namespace std;

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
  
  emptyState = new float[historySize*stateSize];
  for (int i=0; i<historySize*stateSize; i++)
  {
    emptyState[i] = -128.0;
  }
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
    
    //bStates[batchSize][historySize][stateSize]
    
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
      memcpy(&bStates[stateSize*((i+1)*historySize-1-j)], emptyState, sizeof(float)*stateSize);
    }
    
    bActions[i] = actions[sInd];
    bRewards[i] = rewards[sInd];
    bDones[i] = dones[sInd];

    if (dones[sInd] || nextSind == ind)
    {
      memcpy(&bNextStates[i*historySize*stateSize], emptyState, sizeof(float)*historySize*stateSize);
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
  int j=0;
  
  if (!dones[sInd])
  {
    memcpy(&bStates[stateSize*(historySize-2-j)], &states[sInd * stateSize], sizeof(float)*stateSize);
    int curSInd = sInd;
    j++;
  
    for (; (curSInd != 0 || curSize == bufferSize) && curSInd != ind && j<historySize-1; j++)
    {
      curSInd = (curSInd-1+curSize)%curSize;
      if (dones[curSInd])
      {
        break;
      }
      memcpy(&bStates[(historySize-2-j)*stateSize], &states[curSInd*stateSize], sizeof(float)*stateSize);
    }
  }
  for (; j<historySize-1; j++)
  {
    memcpy(&bStates[(historySize-2-j)*stateSize], emptyState, sizeof(float)*stateSize);
  }
  
  memcpy(&bStates[(historySize-1)*stateSize], curState, sizeof(float)*stateSize);
}
/* // Replay buffer test driver, does not compile with MSVCP
int main(int argc, const char* argv)
{
  int stateSize = 3;
  int bufferSize = 5;
  int batchSize = 2;
  int historySize = 3;
  
  int actionRange = 3;
  int rewardRange = 5;
  
  ReplayBuffer replay(stateSize, bufferSize, batchSize, historySize);
  
  for (int i=0; i< 100; i++)
  {
    cout << "ADDED:\n";
  
    float state[3];
    for (int j=0; j<stateSize; j++)
    {
      state[j] = rand()%10;
      cout << state[j] << ", ";
    }
    cout << endl;
    int action = rand()%actionRange;
    float reward = rand()%rewardRange;
    int done = (rand()%5 == 4);
    replay.add(state, action, reward, done);
    
    cout << "action: " << action << endl;
    cout << "reward: " << reward << endl;
    cout << "done: " << done << endl;
    
    if ((i+1)%5 == 0)
    {
      float bStates[batchSize][historySize][stateSize];
      int64_t bActions[batchSize];
      float bRewards[batchSize];
      float bNextStates[batchSize][historySize][stateSize];
      int64_t bDones[batchSize];
      
      float recent[historySize][stateSize];
      
      
      replay.sample(bStates[0][0], bActions, bRewards, bNextStates[0][0], bDones);
      
      
      
      cout << "\n\n\n\n\nstate batch:\n";
      for (int j=0; j<batchSize; j++)
      {
        for (int h=0; h<historySize; h++)
        {
          for (int k=0; k<stateSize; k++)
          {
            cout << bStates[j][h][k] << ", ";
          }
          cout << endl << "," << endl;
        }
        cout << "\n\n\n";
      }
      
      cout << "next state batch:\n";
      for (int j=0; j<batchSize; j++)
      {
        for (int h=0; h<historySize; h++)
        {
          for (int k=0; k<stateSize; k++)
          {
            cout << bNextStates[j][h][k] << ", ";
          }
          cout << endl << "," << endl;
        }
        cout << "\n\n\n";
      }
      
      float temp[stateSize];
      float curState[historySize][stateSize];
      for (int j=0; j<stateSize; j++)
      {
        temp[j] = 1.0f;
      }
      replay.recent(curState[0], temp);
      
      cout << "recent states:\n";
      for (int j=0; j<historySize; j++)
      {
        for (int h=0; h<stateSize; h++)
        {
          cout << curState[j][h] << ", ";
        }
        cout << endl;
      }
      cout << endl;
      
      cout << "action batch:\n";
      for (int j=0; j<batchSize; j++)
      {
        cout << bActions[j] << endl;
      }
      
      cout << "reward batch:\n";
      for (int j=0; j<batchSize; j++)
      {
        cout << bRewards[j] << endl;
      }
      
      cout << "done batch:\n";
      for (int j=0; j<batchSize; j++)
      {
        cout << bDones[j] << endl;
      }
      
      cout << "waiting..." << endl;
      int dummy;
      cin >> dummy;
    }
  }
}
*/

ReplayBuffer::~ReplayBuffer()
{
  delete [] states;
  delete [] actions;
  delete [] rewards;
  delete [] dones;
  delete [] emptyState;
}
