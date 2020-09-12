#include "reinforce.hpp"
#include <stdlib.h>
#include <time.h>
#include <string>

const int layerSize = 50;

Reinforce::Reinforce(int inStateSize, int inActionSize, float policy_lr, float inGamma)
{
  /*if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = new torch::Device(torch::kCUDA);
  }
  else*/
  {
    device = new torch::Device(torch::kCPU);
  }
  
  stateSize = inStateSize;
  actionSize = inActionSize;
  
  gamma = inGamma;
  actionSize = inActionSize;
  
  actorModel = MLP(stateSize, actionSize, layerSize);
  actorModel->to(*device);
  actorModel_optimizer = new torch::optim::Adam(actorModel->parameters(), torch::optim::AdamOptions(policy_lr).eps(1e-3));
  
  kCheckpointEvery = 10000;
  
  itCounter = 0;
  checkpoint_counter = 0;
  
  buffer = new Buffer(stateSize);
  
  srand(time(NULL));
  
  printf("policy_lr %e\n", policy_lr);
  printf("gamma %e\n", gamma);
}

int64_t Reinforce::chooseAction(float* state)
{
  Tensor states = torch::from_blob(state, {1, stateSize}).to(*device);
  Tensor logits = actorModel->forward(states);
  Tensor probs = logits.softmax(1);
  Tensor actionsSamples = probs.multinomial(1);
  
  return actionsSamples.item().toInt();
}

float Reinforce::remember(float* state, int64_t action, float reward, int64_t done)
{  
  buffer->add(state, action, reward);
  
  float fLoss = 0;
  
  if (done)
  {
    buffer->processDiscountedRewards(gamma);
    
    actorModel_optimizer->zero_grad();
    
    Tensor states = torch::from_blob(&buffer->states[0], {buffer->processedSize(), stateSize}).to(*device);
    
    Tensor actions = torch::from_blob(&buffer->actions[0], {buffer->processedSize(), 1}, TensorOptions().dtype(kInt64)).to(*device);
    Tensor ret = torch::from_blob(&buffer->rewardsToGo[0], {buffer->processedSize(), 1}).to(*device);
    Tensor logits = actorModel->forward(states);
    
    Tensor logSoft = logits.log_softmax(1);
    Tensor gathered = logSoft.gather(1, actions);
    Tensor losses = gathered * -ret;
    Tensor loss = losses.mean();
    loss.backward();
    actorModel_optimizer->step();
    fLoss = loss.item<float>();
    
    cout << "BUFFER POPPED" << endl;

    buffer->clearProcessed();
  }
  
  /*if (itCounter % kCheckpointEvery == 0) {
      // Checkpoint the model and optimizer state.
      torch::save(actorModel, "model-checkpoint.pt");
      torch::save(*actorModel_optimizer, "model-optimizer-checkpoint.pt");
  }*/
  
  itCounter++;
  
  return fLoss;
}

Reinforce::~Reinforce()
{
  saveFile.close();
  delete device;
  delete actorModel_optimizer;
  delete buffer;
}

Reinforce* createReinforce(int inStateSize, int inActionSize, float policy_lr, float inGamma)
{
  return new Reinforce(inStateSize, inActionSize, policy_lr, inGamma);
}

void freeReinforce(Reinforce* reinforce)
{
  delete reinforce;
}

int64_t chooseAction(Reinforce* reinforce, float* state)
{
  int64_t result = reinforce->chooseAction(state);
  return result;
}

float remember(Reinforce* reinforce, float* state, int64_t action, float reward, int64_t done)
{
  float result = reinforce->remember(state, action, reward, done);
  return result;
}

extern "C"
{
  typedef struct Reinforce Reinforce;
void* createAgentc(int inStateSize, int inActionSize, float policy_lr, float inGamma)
  {
    return (void*)createReinforce(inStateSize, inActionSize, policy_lr, inGamma);
  }
  
  void freeAgentc(void* reinforce)
  {
    freeReinforce((Reinforce*)reinforce);
  }
  
  int64_t chooseActionc(void* reinforce, float* state)
  {
    return chooseAction((Reinforce*)reinforce, state);
  }
  
  float rememberc(void* reinforce, float* state, int64_t action, float reward, int64_t done)
  {
    return remember((Reinforce*)reinforce, state, action, reward, done);
  }
}
