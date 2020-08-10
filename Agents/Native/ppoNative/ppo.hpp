#ifndef PPO_H
#define PPO_H

#include <vector>
#include <torch/torch.h>
#include <string>
#include <assert.h>
#include <iostream>
#include <fstream>
//#include <tensorboard_logger.h>
#include "buffer.h"

using namespace torch;
using namespace std;

struct CriticImpl : nn::Module
{
  CriticImpl(int stateSize, int layerSize)
  {     
     qlin1 = nn::Linear(nn::LinearOptions(stateSize, layerSize));
     qlin2 = nn::Linear(nn::LinearOptions(layerSize, layerSize));
     qlin3 = nn::Linear(nn::LinearOptions(layerSize, 1));
     
     register_module("qlin1", qlin1);
     register_module("qlin2", qlin2);
     register_module("qlin3", qlin3);
  }

  Tensor forward(Tensor x)
  {    
    x = torch::relu(qlin1(x));
    x = torch::relu(qlin2(x));
    x = qlin3(x);
    
    return x;
  }

  nn::Linear qlin1 = nullptr, qlin2 = nullptr, qlin3 = nullptr;
};

TORCH_MODULE(Critic);

struct ActorImpl : nn::Module
{
  ActorImpl(int stateSize, int outputSize, int layerSize, float inEps)
  {
    eps = inEps;
    
     qlin1 = nn::Linear(nn::LinearOptions(stateSize, layerSize));
     qlin2 = nn::Linear(nn::LinearOptions(layerSize, layerSize));
     qlin3 = nn::Linear(nn::LinearOptions(layerSize, outputSize));
     
     register_module("qlin1", qlin1);
     register_module("qlin2", qlin2);
     register_module("qlin3", qlin3);
 }
  
  Tensor network(Tensor x)
  {
    x = torch::relu(qlin1(x));
    x = torch::relu(qlin2(x));
    x = qlin3(x);
    
    return x;
  }
  
  Tensor probs(Tensor state)
  {
    Tensor logits = network(state);
    Tensor out = logits.softmax(1);
    return out;
  }
  
  Tensor actionLogProbs(Tensor state, Tensor action)
  {
    Tensor logits = network(state);
    Tensor logSoft = logits.log_softmax(1);
    Tensor gathered = logSoft.gather(1, action);
    return gathered;
  }
  
  Tensor forward(Tensor oldGathered, Tensor state, Tensor action, Tensor advantage)
  {
    Tensor newGathered = actionLogProbs(state, action);
    Tensor gathDiff = torch::exp(newGathered - oldGathered);
    Tensor loss = torch::min(gathDiff * advantage, torch::clamp(gathDiff, 1-eps, 1+eps) * advantage).mean();
    loss = -loss;
    return loss;
  }
  
  float eps;
  nn::Linear qlin1 = nullptr, qlin2 = nullptr, qlin3 = nullptr;
};

TORCH_MODULE(Actor);

class PPO
{
  public:
    PPO(int inStateSize, int inActionSize, float policy_lr, float value_lr, float inGamma, int inHorizon, int epoch, int inMiniBatchSize, float inEps, float inLambda);
    ~PPO();
    int64_t chooseAction(float* state);
    float remember(float* state, int64_t action, float reward, int64_t done);
    ofstream saveFile;
    //TensorBoardLogger* logger = nullptr;
  private:
    float getDiscountedRewards();

    torch::Device* device;
    
    Actor actorModel = nullptr;
    Critic criticModel = nullptr;
    
    torch::optim::Adam* actorModel_optimizer;
    torch::optim::Adam* criticModel_optimizer;
    
    Buffer* buffer;
    
    int stateSize;
    int actionSize;
    float gamma;
    int64_t epochSize;
    int64_t horizon;
    int64_t actorIts;
    int64_t criticIts;
    int64_t miniBatchSize;
    float eps;
    float lambda;
    
    int64_t kCheckpointEvery;
    
    int64_t itCounter;
    int64_t checkpoint_counter;
        
    float curValEst;
};

#ifdef _WIN32
extern "C"
{
    __declspec(dllexport) void* createAgentc(int inStateSize, int inActionSize, float policy_lr, float value_lr, float inGamma, int inHorizon, int epoch, int inMiniBatchSize, float inEps, float inLambda);
    __declspec(dllexport) void freeAgentc(void* ppo);
    __declspec(dllexport) int64_t chooseActionc(void* ppo, float* state);
    __declspec(dllexport) float rememberc(void* ppo, float* state, int64_t action, float reward, int64_t done);
}
#endif
#endif
