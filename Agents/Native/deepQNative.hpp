#ifndef DQN_H
#define DQN_H
#include <torch/torch.h>

using namespace torch;

const int64_t stateSize = 4;
const int64_t layerSize = 10;
const int64_t outputSize = 2;

const int64_t bufferSize = 6553600;
const int64_t batchSize = 32;

const bool kRestoreFromCheckpoint = false;
const int64_t kLogInterval = 100;
const int64_t kCheckpointEvery = 1000;
const int64_t kNumberOfSamplesPerCheckpoint = 10;
const int64_t targetUpdate = 100;

/*struct BallTestImpl: nn::Module
{
  BallTestImpl()
      : lin1(nn::LinearOptions(stateSize, layerSize)),
        lin2(nn::LinearOptions(layerSize, layerSize)),
        lin3(nn::LinearOptions(layerSize, layerSize)),
        lin4(nn::LinearOptions(layerSize, outputSize))
 {
   // register_module() is needed if we want to use the parameters() method later on
   register_module("lin1", lin1);
   register_module("lin2", lin2);
   register_module("lin3", lin3);
   register_module("lin4", lin4);
 }

 torch::Tensor forward(torch::Tensor x, torch::Tensor mask) {
   x = torch::relu(lin1(x));
   x = torch::relu(lin2(x));
   x = torch::relu(lin3(x));
   x = mask * lin4(x) ;
   
   return x;
 }

 void memload() {
   
 }

 nn::Linear lin1, lin2, lin3, lin4;
};

TORCH_MODULE(BallTest);*/

struct DuelingBallTestImpl: nn::Module
{
  DuelingBallTestImpl() : lin1(nn::LinearOptions(stateSize, layerSize)),
    lin2(nn::LinearOptions(layerSize, layerSize)),
    qlin1(nn::LinearOptions(layerSize, layerSize)),
    qlin2(nn::LinearOptions(layerSize, outputSize)),
    aaqlin1(nn::LinearOptions(layerSize, layerSize)),
    aaqlin2(nn::LinearOptions(layerSize, outputSize))
  {
    register_module("lin1", lin1);
    register_module("lin2", lin2);
    register_module("qlin1", qlin1);
    register_module("qlin2", qlin2);
    register_module("aaqlin1", aaqlin1);
    register_module("aaqlin2", aaqlin2);
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor mask) {
    x = torch::relu(lin1(x));
    x = torch::relu(lin2(x));
    Tensor q = torch::relu(qlin1(x));
    q = qlin2(q);
    Tensor aaq = torch::relu(aaqlin1(x));
    aaq = aaqlin2(aaq);

    return mask * ((q + aaq) - aaq.mean(1, true));
  }

  void memload() {
   
  }

  nn::Linear lin1, lin2, qlin1, qlin2, aaqlin1, aaqlin2;
};

TORCH_MODULE(DuelingBallTest);

struct ReplayBuffer
{
  ReplayBuffer() : curSize(0), ind(0) {
    srand(time(NULL));
  }

  void add(float* state, int64_t action, float reward, int64_t done)
  {
    memcpy(&states[ind], state, sizeof(float)*stateSize);
    actions[ind] = action;
    rewards[ind] = reward;
    dones[ind] = done;
    ind = (ind+1)%bufferSize;
    if (curSize < bufferSize)
    {
      curSize++;
    }
  }

  void sample(float (*bStates)[stateSize], int64_t* bActions, float* bRewards, float (*bNextStates)[stateSize], int64_t* bDones)
  {
    for (int i=0; i<batchSize; i++)
    {
      int sInd = rand()%curSize;
      int nextSind = (sInd+1)%bufferSize;
      memcpy(&bStates[i], &states[sInd], sizeof(float)*stateSize);
      bActions[i] = actions[sInd];
      bRewards[i] = rewards[sInd];
      bDones[i] = dones[sInd];
      if (bDones[i] || nextSind == ind)
      {
        memset(&bNextStates[i], 0, sizeof(float)*stateSize);
      }
      else
      {
        memcpy(&bNextStates[i], &states[nextSind], sizeof(float)*stateSize);
      }
    }
  }

  float states[bufferSize][stateSize];
  int64_t actions[bufferSize];
  float rewards[bufferSize];
  int64_t dones[bufferSize];
  int curSize;
  int ind;
};

class DQN
{
    public:
        DQN();
        ~DQN();
        int64_t chooseAction(float state[stateSize]);
        float remember(float state[stateSize], int64_t action, float reward, int64_t done);
        void newEpisode(){}
        
    private:
        torch::Device* device;
        DuelingBallTest model;
        DuelingBallTest target;
        torch::optim::Adam* model_optimizer;
        Tensor fullMask;
        ReplayBuffer* replay;
        
        float gamma;
        float epsilon;
        float decayRate;
        int64_t itCounter;
        int64_t checkpoint_counter;
};

#endif
