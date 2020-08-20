#ifndef ACTORCRITIC_H
#define ACTORCRITIC_H

#include <vector>
#include <torch/torch.h>
#include <string>
#include <assert.h>
#include <iostream>
#include <fstream>
#include "buffer.h"

using namespace torch;
using namespace std;

struct MLPImpl : nn::Module
{
  MLPImpl(int stateSize, int outputSize, int layerSize)
  {    
     lin1 = nn::Linear(nn::LinearOptions(stateSize, layerSize));
     lin2 = nn::Linear(nn::LinearOptions(layerSize, layerSize));
     lin3 = nn::Linear(nn::LinearOptions(layerSize, outputSize));
     
     register_module("lin1", lin1);
     register_module("lin2", lin2);
     register_module("lin3", lin3);
 }
  
  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(lin1(x));
    x = torch::relu(lin2(x));
    x = lin3(x);
   
   return x;
  }
  
 nn::Linear lin1 = nullptr, lin2 = nullptr, lin3 = nullptr;
};

TORCH_MODULE(MLP);

class ActorCritic
{
  public:
    ActorCritic(int inStateSize, int inActionSize, float policy_lr, float value_lr, float inGamma, int inHorizon, int epoch);
    ~ActorCritic();
    int64_t chooseAction(float* state);
    float remember(float* state, int64_t action, float reward, int64_t done);
    ofstream saveFile;
  private:
    float getDiscountedRewards();

    torch::Device* device;
    
    MLP actorModel = nullptr;
    MLP valueModel = nullptr;
    
    torch::optim::Adam* actorModel_optimizer;
    torch::optim::Adam* valueModel_optimizer;
    
    Buffer* buffer;
    
    int stateSize;
    int actionSize;
    float gamma;
    int64_t epochSize;
    int64_t horizon;
    int64_t actorIts;
    int64_t criticIts;
    int64_t miniBatchSize;
    float lambda;
    
    int64_t kCheckpointEvery;
    
    int64_t itCounter;
    int64_t checkpoint_counter;
        
    float curValEst;
};

#ifdef _WIN32
extern "C"
{
    __declspec(dllexport) void* createAgentc(int inStateSize, int inActionSize, float policy_lr, float value_lr, float inGamma, int inHorizon, int epoch);
    __declspec(dllexport) void freeAgentc(void* actorCritic);
    __declspec(dllexport) int64_t chooseActionc(void* actorCritic, float* state);
    __declspec(dllexport) float rememberc(void* actorCritic, float* state, int64_t action, float reward, int64_t done);
}
#endif
#endif
