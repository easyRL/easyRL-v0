#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>
#include "deepQNative.hpp"

using namespace std;

DQN::DQN()
{
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = new torch::Device(torch::kCUDA);
  }
  else
  {
    device = new torch::Device(torch::kCPU);
  }

  model = DuelingBallTest();
  model->to(*device);
  target = DuelingBallTest();
  target->to(*device);
  
  model_optimizer = new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(7e-4));

  fullMask = torch::ones({1,outputSize}).to(*device);
  
  replay = new ReplayBuffer();
  printf("BUFFERSIZE: %lu\n", sizeof(ReplayBuffer));
  
  gamma = 0.99f;
  epsilon = 1.0f;
  decayRate = 1.0f/15000.0f;
  itCounter = 0;
  checkpoint_counter = 0;

  cout << "Learning rate: " << 1e-3 << ", gamma: " << gamma << ", target update rate: " << targetUpdate << endl;
}

int64_t DQN::chooseAction(float state[stateSize])
{
  int64_t action;
  Tensor xsingle = torch::from_blob(state, {1, stateSize}).to(*device);
  Tensor ysingle = model->forward(xsingle, fullMask);
  action = ysingle.argmax(1).item().toInt();
    //action = rand()%outputSize;
    //std::cout << "ACTION " << action << " RANDOMED" << std::endl;
  return action;
}

float DQN::remember(float state[stateSize], int64_t action, float reward, int64_t done)
{
  float fLoss=0;
  model_optimizer->zero_grad();
  replay->add(state, action, reward, done);
  
  if (replay->curSize >= batchSize)
  {  
    float bStates[batchSize][stateSize];
    int64_t bActions[batchSize];
    float bRewards[batchSize];
    float bNextStates[batchSize][stateSize];
    int64_t bDones[batchSize];
    
    replay->sample(bStates, bActions, bRewards, bNextStates, bDones);
    
    Tensor xbatch = torch::from_blob(bStates, {batchSize, stateSize}).to(*device);        
    Tensor actionsbatch = torch::from_blob(bActions, {batchSize, 1}, TensorOptions().dtype(kInt64)).to(*device);
    Tensor rewardsbatch = torch::from_blob(bRewards, {batchSize, 1}).to(*device);
    Tensor nextxbatch = torch::from_blob(bNextStates, {batchSize, stateSize}).to(*device);
    Tensor donesbatch = torch::from_blob(bDones, {batchSize, 1}, TensorOptions().dtype(kInt64)).to(*device);
    
    Tensor actionsOneHotBatch = (torch::zeros({batchSize, outputSize}).to(*device).scatter_(1, actionsbatch, 1)).to(*device);
    Tensor ybatch = model->forward(xbatch, actionsOneHotBatch);
    Tensor nextybatch = model->forward(nextxbatch, fullMask);
    Tensor nextybatchTarg = target->forward(nextxbatch, fullMask);
    Tensor argmaxes = nextybatch.argmax(1, true);
    Tensor maxes = nextybatchTarg.gather(1, argmaxes);
    Tensor nextvals = rewardsbatch + (1 - donesbatch) * (gamma * maxes);    
    
    Tensor targetbatch = torch::zeros({batchSize, outputSize}).to(*device).scatter_(1, actionsbatch, nextvals);    
    
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
    
    if (itCounter % kCheckpointEvery == 0) {
      // Checkpoint the model and optimizer state.
      torch::save(model, "model-checkpoint.pt");
      torch::save(*model_optimizer, "model-optimizer-checkpoint.pt");
      std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
    }
    
    itCounter++;
  }
  
  if (done)
  {
    std::printf("                             Epsilon %f\n", epsilon);
  }
  
  return fLoss;
}

DQN::~DQN()
{
  delete device;
  delete model_optimizer;
  delete replay;
}

DQN* createDQN()
{
  return new DQN;
}

void freeDQN(DQN* dqn)
{
  delete dqn;
}

int64_t chooseAction(DQN* dqn, float state[stateSize])
{
  int64_t result = dqn->chooseAction(state);
  return result;
}

float remember(DQN* dqn, float state[stateSize], int64_t action, float reward, int64_t done)
{
  float result = dqn->remember(state, action, reward, done);
  return result;
}

extern "C"
{
  typedef struct DQN DQN;
  DQN* createDQNc()
  {
    return createDQN();
  }
  
  void freeDQNc(DQN* dqn)
  {
    freeDQN(dqn);
  }
  
  int64_t chooseActionc(DQN* dqn, float state[stateSize])
  {
    return chooseAction(dqn, state);
  }
  
  float rememberc(DQN* dqn, float state[stateSize], int64_t action, float reward, int64_t done)
  {
    return remember(dqn, state, action, reward, done);
  }
}
