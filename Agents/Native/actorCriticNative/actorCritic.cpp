#include "actorCritic.hpp"
#include <stdlib.h>
#include <time.h>
#include <string>

const int layerSize = 50;

ActorCritic::ActorCritic(int inStateSize, int inActionSize, float policy_lr, float value_lr, float inGamma, int inHorizon, int epoch)
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
  epochSize = epoch;
  horizon = inHorizon;
  actionSize = inActionSize;
  
  actorModel = MLP(stateSize, actionSize, layerSize);
  actorModel->to(*device);
  actorModel_optimizer = new torch::optim::Adam(actorModel->parameters(), torch::optim::AdamOptions(policy_lr).eps(1e-3));
  
  valueModel = MLP(stateSize, 1, layerSize);
  valueModel->to(*device);
  valueModel_optimizer = new torch::optim::Adam(actorModel->parameters(), torch::optim::AdamOptions(policy_lr).eps(1e-3));
  
  kCheckpointEvery = 10000;
  
  itCounter = 0;
  checkpoint_counter = 0;
  
  buffer = new Buffer(stateSize);
  
  srand(time(NULL));
  
  /*torch::load(model, "model-checkpoint.pt");
  torch::load(*model_optimizer, "model-optimizer-checkpoint.pt");
  torch::load(value, "value-checkpoint.pt");
  torch::load(*value_optimizer, "value-optimizer-checkpoint.pt");*/
  
  printf("policy_lr %e\n", policy_lr);
  printf("value_lr %e\n", value_lr);
  printf("gamma %e\n", gamma);
  printf("horizon %d\n", horizon);
  printf("epoch size%d\n", epochSize);
}

int64_t ActorCritic::chooseAction(float* state)
{
  Tensor states = torch::from_blob(state, {1, stateSize}).to(*device);
  Tensor logits = actorModel->forward(states);
  Tensor probs = logits.softmax(1);
  curValEst = valueModel->forward(states).item<float>();
  Tensor actionsSamples = probs.multinomial(1);
  
  return actionsSamples.item().toInt();
}

float ActorCritic::remember(float* state, int64_t action, float reward, int64_t done)
{  
  buffer->add(state, action, reward, curValEst, done);
  
  float fLoss = 0;
  
  if (done || (buffer->unProcessedSize() >= horizon))
  {
    buffer->processDiscountedRewards(gamma, done);
  }
  
  if (buffer->storedSize() >= epochSize)
  {
    buffer->processDiscountedRewards(gamma, done);
    
    actorModel_optimizer->zero_grad();
    valueModel_optimizer->zero_grad();
    
    Tensor states = torch::from_blob(&buffer->states[0], {buffer->processedSize(), stateSize}).to(*device);
    
    Tensor actions = torch::from_blob(&buffer->actions[0], {buffer->processedSize(), 1}, TensorOptions().dtype(kInt64)).to(*device);
    Tensor ret = torch::from_blob(&buffer->returns[0], {buffer->processedSize(), 1}).to(*device);
    Tensor rtg = torch::from_blob(&buffer->rewardsToGo[0], {buffer->processedSize(), 1}).to(*device);
    Tensor logits = actorModel->forward(states);
    
    Tensor logSoft = logits.log_softmax(1);
    Tensor gathered = logSoft.gather(1, actions);
    Tensor losses = gathered * -ret;
    Tensor loss = losses.mean();
    loss.backward();
    actorModel_optimizer->step();
    fLoss = loss.item<float>();
    
    Tensor vals = valueModel->forward(states);
    Tensor val_loss = torch::mse_loss(vals, rtg);
    val_loss.backward();
    valueModel_optimizer->step();
    
    cout << "BUFFER POPPED" << endl;

    buffer->clearProcessed();
  }
  
  /*if (itCounter % kCheckpointEvery == 0) {
      // Checkpoint the model and optimizer state.
      torch::save(actorModel, "model-checkpoint.pt");
      torch::save(*actorModel_optimizer, "model-optimizer-checkpoint.pt");
      
      torch::save(valueModel, "value-checkpoint.pt");
      torch::save(*valueModel_optimizer, "value-optimizer-checkpoint.pt");
      std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
  }*/
  
  itCounter++;
  
  return fLoss;
}

ActorCritic::~ActorCritic()
{
  saveFile.close();
  delete device;
  delete actorModel_optimizer;
  delete valueModel_optimizer;
  delete buffer;
}

ActorCritic* createActorCritic(int inStateSize, int inActionSize, float policy_lr, float value_lr, float inGamma, int inHorizon, int epoch)
{
  return new ActorCritic(inStateSize, inActionSize, policy_lr, value_lr, inGamma, inHorizon, epoch);
}

void freeActorCritic(ActorCritic* actorCritic)
{
  delete actorCritic;
}

int64_t chooseAction(ActorCritic* actorCritic, float* state)
{
  int64_t result = actorCritic->chooseAction(state);
  return result;
}

float remember(ActorCritic* actorCritic, float* state, int64_t action, float reward, int64_t done)
{
  float result = actorCritic->remember(state, action, reward, done);
  return result;
}

extern "C"
{
  typedef struct ActorCritic ActorCritic;
void* createAgentc(int inStateSize, int inActionSize, float policy_lr, float value_lr, float inGamma, int inHorizon, int epoch)
  {
    return (void*)createActorCritic(inStateSize, inActionSize, policy_lr, value_lr, inGamma, inHorizon, epoch);
  }
  
  void freeAgentc(void* actorCritic)
  {
    freeActorCritic((ActorCritic*)actorCritic);
  }
  
  int64_t chooseActionc(void* actorCritic, float* state)
  {
    return chooseAction((ActorCritic*)actorCritic, state);
  }
  
  float rememberc(void* actorCritic, float* state, int64_t action, float reward, int64_t done)
  {
    return remember((ActorCritic*)actorCritic, state, action, reward, done);
  }
}
