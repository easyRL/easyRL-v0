#include <vector>
#include <stdlib.h>

using namespace std;

struct Buffer
{
  Buffer(int inStateSize);

  void add(float* state, int64_t action, float reward, float valEst, int64_t done);
  void processDiscountedRewards(float gamma, float lambda, bool done);

  bool getPrevState(float* prevState);
  bool getPrevAction(int64_t& prevAction);
  void incrementLastReward(float rewardIncrement);
  float getLastReward();
  int storedSize();
  int processedSize();
  int unProcessedSize();
  void clearProcessed();

  vector<float> states;
  vector<int64_t> actions;
  vector<float> rewards;
  vector<float> rewardsToGo;
  vector<float> advantages;
  vector<float> returns;
  vector<int64_t> dones;
  
  int stateSize;
};
