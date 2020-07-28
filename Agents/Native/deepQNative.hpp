#ifndef DQN_H
#define DQN_H
#include <torch/torch.h>

using namespace torch;

struct DuelingImpl: nn::Module
{
  DuelingImpl(int stateSize, int outputSize, int layerSize) : 
    lin1(nn::LinearOptions(stateSize, layerSize)),
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

  nn::Linear lin1, lin2, qlin1, qlin2, aaqlin1, aaqlin2;
};

TORCH_MODULE(Dueling);

struct ReplayBuffer
{
  ReplayBuffer(int inStateSize, int inBufferSize, int inBatchSize) : curSize(0), ind(0) {
    stateSize = inStateSize;
    bufferSize = inBufferSize;
    batchSize = inBatchSize;
  
    srand(time(NULL));
    states = new float[bufferSize * stateSize];
    actions = new int64_t[bufferSize];
    rewards = new float[bufferSize];
    dones = new int64_t[bufferSize];
  }

  void add(float* state, int64_t action, float reward, int64_t done)
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

  void sample(float* bStates, int64_t* bActions, float* bRewards, float* bNextStates, int64_t* bDones)
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
        memset(&bNextStates[i*stateSize], 0.0f, sizeof(float)*stateSize);
      }
      else
      {
        memcpy(&bNextStates[i*stateSize], &states[nextSind*stateSize], sizeof(float)*stateSize);
      }
    }
  }

  ~ReplayBuffer()
  {
    delete [] states;
    delete [] actions;
    delete [] rewards;
    delete [] dones;
  }

  int stateSize;
  int bufferSize;
  int batchSize;
  
  float* states;
  int64_t* actions;
  float* rewards;
  int64_t* dones;
  int curSize;
  int ind;
};

class DQN
{
    public:
        DQN(int stateSize, int actionSize, float gamma, int inBatchSize, int inMemorySize, int inTargetUpdate);
        ~DQN();
        int64_t chooseAction(float* state);
        float remember(float* state, int64_t action, float reward, int64_t done);
        void newEpisode(){}
        void save(char* filename);
        void load(char* filename);
        std::stringstream* memsave();
        void memload(std::stringstream* mem);
        
    private:
        torch::Device* device;
        Dueling model = nullptr;
        Dueling target = nullptr;
        torch::optim::Adam* model_optimizer;
        Tensor fullMask;
        ReplayBuffer* replay;
        
        int stateSize;
        int actionSize;
        float gamma;
        int batchSize;
        int memorySize;
        int targetUpdate;
        int64_t itCounter;
        int64_t checkpoint_counter;
};

#endif
