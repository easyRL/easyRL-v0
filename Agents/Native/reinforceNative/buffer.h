#include <vector>
#include <stdlib.h>

using namespace std;

struct Buffer
{
  Buffer(int inStateSize);

  void add(float* state, int64_t action, float reward);
  void processDiscountedRewards(float gamma);

  bool getPrevState(float* prevState);
  bool getPrevAction(int64_t& prevAction);
  float getLastReward();
  int storedSize();
  int processedSize();
  int unProcessedSize();
  void clearProcessed();

  vector<float> states;
  vector<int64_t> actions;
  vector<float> rewards;
  vector<float> rewardsToGo;
  
  int stateSize;
};
