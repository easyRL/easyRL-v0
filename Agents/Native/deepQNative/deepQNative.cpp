#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "deepQNative.hpp"

using namespace std;

const int layerSize = 10;

DQN::DQN(int inStateSize, int inActionSize, float inGamma, int inBatchSize, int inMemorySize, int inTargetUpdate, float learningRate)
{
  stateSize = inStateSize;
  actionSize = inActionSize;
  gamma = inGamma;
  batchSize = inBatchSize;
  memorySize = inMemorySize;
  targetUpdate = inTargetUpdate;

  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = new torch::Device(torch::kCUDA);
  }
  else
  {
    device = new torch::Device(torch::kCPU);
  }

  model = Dueling(stateSize, actionSize, layerSize);
  model->to(*device);
  target = Dueling(stateSize, actionSize, layerSize);
  target->to(*device);
  
  model_optimizer = new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(learningRate));

  fullMask = torch::ones({1,actionSize}).to(*device);
  
  replay = new ReplayBuffer(stateSize, memorySize, batchSize);
  printf("BUFFERSIZE: %lu\n", sizeof(ReplayBuffer));
  
  gamma = inGamma;
  itCounter = 0;
  checkpoint_counter = 0;

  cout << "Learning rate: " << learningRate << ", target update rate: " << targetUpdate << endl;
  cout << "stateSize: " << stateSize << ", actionSize: " << actionSize << ", gamma: " << gamma << endl;
  cout << ", batchSize: " << batchSize << ", memorySize: " << memorySize << ", targetUpdate: " << targetUpdate << endl;
}

int64_t DQN::chooseAction(float* state)
{
  int64_t action;
  Tensor xsingle = torch::from_blob(state, {1, stateSize}).to(*device);
  Tensor ysingle = model->forward(xsingle, fullMask);
  action = ysingle.argmax(1).item().toInt();
    //action = rand()%outputSize;
    //std::cout << "ACTION " << action << " RANDOMED" << std::endl;
  return action;
}

float DQN::remember(float* state, int64_t action, float reward, int64_t done)
{
  float fLoss=0;
  model_optimizer->zero_grad();
  replay->add(state, action, reward, done);
  
  
  if (replay->curSize >= batchSize)
  {  
    float* bStates = new float[batchSize * stateSize];
    int64_t* bActions = new int64_t[batchSize];
    float* bRewards = new float[batchSize];
    float* bNextStates = new float[batchSize * stateSize];
    int64_t* bDones = new int64_t[batchSize];
    
    replay->sample(bStates, bActions, bRewards, bNextStates, bDones);
    
    Tensor xbatch = torch::from_blob(bStates, {batchSize, stateSize}).to(*device);        
    Tensor actionsbatch = torch::from_blob(bActions, {batchSize, 1}, TensorOptions().dtype(kInt64)).to(*device);
    Tensor rewardsbatch = torch::from_blob(bRewards, {batchSize, 1}).to(*device);
    Tensor nextxbatch = torch::from_blob(bNextStates, {batchSize, stateSize}).to(*device);
    Tensor donesbatch = torch::from_blob(bDones, {batchSize, 1}, TensorOptions().dtype(kInt64)).to(*device);
    
    Tensor actionsOneHotBatch = (torch::zeros({batchSize, actionSize}).to(*device).scatter_(1, actionsbatch, 1)).to(*device);
    Tensor ybatch = model->forward(xbatch, actionsOneHotBatch);
    Tensor nextybatch = model->forward(nextxbatch, fullMask);
    Tensor nextybatchTarg = target->forward(nextxbatch, fullMask);
    Tensor argmaxes = nextybatch.argmax(1, true);
    Tensor maxes = nextybatchTarg.gather(1, argmaxes);
    Tensor nextvals = rewardsbatch + (1 - donesbatch) * (gamma * maxes);    
    
    Tensor targetbatch = torch::zeros({batchSize, actionSize}).to(*device).scatter_(1, actionsbatch, nextvals);
    
    torch::Tensor loss = torch::mse_loss(ybatch, targetbatch.detach());
    loss.backward();
    model_optimizer->step();
    fLoss = loss.item<float>();
    
    if ((itCounter+1) % targetUpdate == 0)
    {
      std::stringstream stream;
      torch::save(model, stream);
      torch::load(target, stream);
      std::cout << "target updated" << std::endl;
    }
    
    /*if (itCounter % kCheckpointEvery == 0) {
      // Checkpoint the model and optimizer state.
      torch::save(model, "model-checkpoint.pt");
      torch::save(*model_optimizer, "model-optimizer-checkpoint.pt");
      std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
    }*/
    
    itCounter++;
    
    delete [] bStates;
    delete [] bActions;
    delete [] bRewards;
    delete [] bNextStates;
    delete [] bDones;
  }
  
  return fLoss;
}

void DQN::save(char* filename)
{
  torch::save(model, filename);
}

void DQN::load(char* filename)
{
  torch::load(model, filename);
}

std::stringstream* DQN::memsave()
{
  std::stringstream* mem = new std::stringstream;
  torch::save(model, *mem);
  return mem;
}

void DQN::memload(std::stringstream* mem)
{
  torch::load(model, *mem);
  delete mem;
}

DQN::~DQN()
{
  delete device;
  delete model_optimizer;
  delete replay;
}

DQN* createDQN(int stateSize, int actionSize, float gamma, int inBatchSize, int inMemorySize, int inTargetUpdate, float learningRate)
{
  return new DQN(stateSize, actionSize, gamma, inBatchSize, inMemorySize, inTargetUpdate, learningRate);
}

void freeDQN(DQN* dqn)
{
  delete dqn;
}

int64_t chooseAction(DQN* dqn, float* state)
{
  int64_t result = dqn->chooseAction(state);
  return result;
}

float remember(DQN* dqn, float* state, int64_t action, float reward, int64_t done)
{
  float result = dqn->remember(state, action, reward, done);
  return result;
}

void save(DQN* dqn, char* filename)
{
  dqn->save(filename);
}

void load(DQN* dqn, char* filename)
{
  dqn->load(filename);
}

void* memsave(DQN* dqn)
{
  return (void*)dqn->memsave();
}

void memload(DQN* dqn, void* mem)
{
  dqn->memload((std::stringstream*)mem);
}

extern "C"
{
  typedef struct DQN DQN;
  void* createAgentc(int stateSize, int actionSize, float gamma, int inBatchSize, int inMemorySize, int inTargetUpdate, float learningRate)
  {
    return (void*)createDQN(stateSize, actionSize, gamma, inBatchSize, inMemorySize, inTargetUpdate, learningRate);
  }
  
  void freeAgentc(void* dqn)
  {
    freeDQN((DQN*)dqn);
  }
  
  int64_t chooseActionc(void* dqn, float* state)
  {
    return chooseAction((DQN*)dqn, state);
  }
  
  float rememberc(void* dqn, float* state, int64_t action, float reward, int64_t done)
  {
    return remember((DQN*)dqn, state, action, reward, done);
  }
  
  void savec(void* dqn, char* filename)
  {
    save((DQN*)dqn, filename);
  }
  
  void loadc(void* dqn, char* filename)
  {
    load((DQN*)dqn, filename);
  }
  
  void* memsavec(void* dqn)
  {
    return memsave((DQN*)dqn);
  }
  
  void memloadc(void* dqn, void* mem)
  {
    memload((DQN*)dqn, mem);
  }
}
