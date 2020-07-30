#include <cstdint>

struct ReplayBuffer
{
  ReplayBuffer(int inStateSize, int inBufferSize, int inBatchSize, int inHistorySize);
  void add(float* state, int64_t action, float reward, int64_t done);
  void sample(float* bStates, int64_t* bActions, float* bRewards,   float* bNextStates, int64_t* bDones);
  void recent(float* bStates, float* curState);
  ~ReplayBuffer();

  int stateSize;
  int bufferSize;
  int batchSize;
  int historySize;
  
  float* states;
  int64_t* actions;
  float* rewards;
  int64_t* dones;
  int curSize;
  int ind;
};
