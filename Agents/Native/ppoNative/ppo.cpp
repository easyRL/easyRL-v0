#include "ppo.hpp"
#include <stdlib.h>
#include <time.h>
#include <string>

const int layerSize = 50;

PPO::PPO(int inStateSize, int inActionSize, float policy_lr, float value_lr, float inGamma, int inHorizon, int epoch, int inMiniBatchSize, float inEps, float inLambda)
{
  //GOOGLE_PROTOBUF_VERIFY_VERSION;

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
  actorIts = 9;
  criticIts = 9;
  miniBatchSize = inMiniBatchSize;
  eps = inEps;
  lambda = inLambda;
  actionSize = inActionSize;
  
  cout << "gamma: " << gamma << endl;
  cout << "epochSize: " << epochSize << endl;
  cout << "horizon: " << horizon << endl;
  cout << "miniBatchSize: " << miniBatchSize << endl;
  cout << "eps: " << eps << endl;
  cout << "lambda: " << lambda << endl;
  cout << "actionSize: " << actionSize << endl;
  
  actorModel = Actor(stateSize, actionSize, layerSize, eps);
  actorModel->to(*device);
  actorModel_optimizer = new torch::optim::Adam(actorModel->parameters(), torch::optim::AdamOptions(policy_lr));
  
  criticModel = Critic(stateSize, layerSize);
  criticModel->to(*device);
  criticModel_optimizer = new torch::optim::Adam(criticModel->parameters(), torch::optim::AdamOptions(value_lr));
  
  //remove("logs/tfevents.pb");
  //logger = new TensorBoardLogger("logs/tfevents.pb");
  
  //saveFile.open("resultsFile"+to_string(param)+".csv", ios::trunc);
  
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
  printf("miniBatchSize %d\n", miniBatchSize);
  printf("eps %e\n", eps);
  printf("lambda %e\n", lambda);
  
  /*auto items = actorModel->named_parameters().items();
  for (auto item : items)
  {
    auto key = item.key();
    auto val = item.value();
    int len = 1;
    for (int i=0; i<val.sizes().size(); i++)
    {
      len *= val.sizes()[i];
    }
    const int clen = len;
    val = val.view({clen});
    assert(val.is_contiguous());
    auto weight_a = val.accessor<float, 1>();
    //cout << key << " added\n\n\n";
    logger->add_histogram(key+".init", itCounter, &weight_a[0], len);
  }*/
}

int64_t PPO::chooseAction(float* state)
{
  Tensor states = torch::from_blob(state, {1, stateSize}).to(*device);
  Tensor probs = actorModel->probs(states);
  curValEst = criticModel->forward(states).item<float>();
  Tensor actionsSamples = probs.multinomial(1);
  return actionsSamples.item().toInt();
}

float PPO::remember(float* state, int64_t action, float reward, int64_t done)
{
  /*cout << "STATE: " << endl;
  for (int i=0; i<stateSize; i++)
  {
    cout << state[i] << ", ";
  }
  cout << "\n\n";*/
  
  buffer->add(state, action, reward, curValEst, done);
  
  float fLoss = 0;
  
  if (done || (buffer->unProcessedSize() >= horizon))
  {
    buffer->processDiscountedRewards(gamma, lambda, done);
  }
  
  if (buffer->storedSize() >= epochSize)
  {  
    buffer->processDiscountedRewards(gamma, lambda, done);
    
    criticModel_optimizer->zero_grad();
    
    Tensor states = torch::from_blob(&buffer->states[0], {buffer->processedSize(), stateSize}).to(*device);
    
    /*cout << "STATE TENSOR:\n" << states << endl;
    int dummy;
    cin >> dummy;*/
    
    Tensor actions = torch::from_blob(&buffer->actions[0], {buffer->processedSize(), 1}, TensorOptions().dtype(kInt64)).to(*device);
    Tensor adv = torch::from_blob(&buffer->advantages[0], {buffer->processedSize(), 1}).to(*device);
    Tensor vals = torch::from_blob(&buffer->returns[0], {buffer->processedSize(), 1}).to(*device);
    Tensor rtg = torch::from_blob(&buffer->rewardsToGo[0], {buffer->processedSize(), 1}).to(*device);
    Tensor dones = torch::from_blob(&buffer->dones[0], {buffer->processedSize(), 1}, TensorOptions().dtype(kInt64)).to(*device);
    
    Tensor gathered = actorModel->actionLogProbs(states, actions);
    
    float totalLoss = 0;
    
    for (int i=0; i<actorIts; i++)
    {
      float accumLoss = 0;
      Tensor shuffleInds = torch::randperm(buffer->processedSize()-1).toType(kInt64).to(*device);
      Tensor nextShuffleInds = shuffleInds + 1;
      int64_t numChunks = (buffer->processedSize() / miniBatchSize);
      if (buffer->processedSize() % miniBatchSize != 0)
      {
        numChunks++;
      }
      vector<Tensor> gathBatch = gathered.detach().index_select(0, shuffleInds).chunk(numChunks);
      vector<Tensor> statesBatch = states.detach().index_select(0, shuffleInds).chunk(numChunks);
      vector<Tensor> actionsBatch = actions.detach().index_select(0, shuffleInds).chunk(numChunks);
      vector<Tensor> advBatch = adv.detach().index_select(0, shuffleInds).chunk(numChunks);
      //vector<Tensor> donesBatch = dones.detach().index_select(0, shuffleInds).chunk(numChunks);
      vector<Tensor> nextStatesBatch = states.detach().index_select(0, nextShuffleInds).chunk(numChunks);
      for (int j=0; j<numChunks; j++)
      {
        actorModel_optimizer->zero_grad();
        Tensor loss = actorModel->forward(gathBatch[j], statesBatch[j], actionsBatch[j], advBatch[j]);
        accumLoss += loss.item<float>();
        loss.backward();
        actorModel_optimizer->step();
      }
      totalLoss += accumLoss / numChunks;
    }
    totalLoss /= criticIts;
    //logger->add_scalar("policy loss", itCounter, totalLoss);
    
    totalLoss = 0;
    for (int i=0; i<criticIts; i++)
    {
      float accumLoss = 0;
      Tensor shuffleInds = torch::randperm(buffer->processedSize()).toType(kInt64).to(*device);
      int64_t numChunks = (buffer->processedSize() / miniBatchSize);
      if (buffer->processedSize() % miniBatchSize != 0)
      {
        numChunks++;
      }
      vector<Tensor> statesBatch = states.detach().index_select(0, shuffleInds).chunk(numChunks);
      vector<Tensor> rtgBatch = rtg.detach().index_select(0, shuffleInds).chunk(numChunks);
      for (int j=0; j<numChunks; j++)
      {
        criticModel_optimizer->zero_grad();
        Tensor predVals = criticModel->forward(statesBatch[j]);
        Tensor loss = torch::mse_loss(predVals, rtgBatch[j]);
        accumLoss += loss.item<float>();
        loss.backward();
        criticModel_optimizer->step();
      }
      totalLoss += accumLoss / numChunks;
    }
    
    totalLoss /= criticIts;
    //logger->add_scalar("value loss", itCounter, totalLoss);
    
    
    /*auto items = actorModel->named_parameters().items();
    for (auto item : items)
    {
      auto key = item.key();
      auto val = item.value();
      auto gradVal = val.grad();
      int len = 1;
      for (int i=0; i<val.sizes().size(); i++)
      {
        len *= val.sizes()[i];
      }
      const int clen = len;
      val = val.view({clen});
      gradVal = gradVal.view({clen});
      assert(val.is_contiguous());
      assert(gradVal.is_contiguous());
      auto weight_a = val.accessor<float, 1>();
      auto grad_a = gradVal.accessor<float, 1>();
      //cout << key << " added\n\n\n";
      logger->add_histogram(key, itCounter, &weight_a[0], len);
      logger->add_histogram(key+".grad", itCounter, &grad_a[0], len);
    }*/
    
    /*items = criticModel->named_parameters().items();
    for (auto item : items)
    {
      auto key = item.key();
      auto val = item.value();
      auto gradVal = val.grad();
      int len = 1;
      for (int i=0; i<val.sizes().size(); i++)
      {
        len *= val.sizes()[i];
      }
      const int clen = len;
      val = val.view({clen});
      gradVal = gradVal.view({clen});
      assert(val.is_contiguous());
      assert(gradVal.is_contiguous());
      auto weight_a = val.accessor<float, 1>();
      auto grad_a = gradVal.accessor<float, 1>();
      //cout << key << " added\n\n\n";
      logger->add_histogram(key, itCounter, &weight_a[0], len);
      logger->add_histogram(key+".grad", itCounter, &grad_a[0], len);
    }*/

    //cout << "BUFFER POPPED\n\n\n" << endl;
    
    buffer->clearProcessed();
  }
  
  /*if (itCounter % kCheckpointEvery == 0) {
      // Checkpoint the model and optimizer state.
      torch::save(actorModel, "model-checkpoint.pt");
      torch::save(*actorModel_optimizer, "model-optimizer-checkpoint.pt");
      
      torch::save(criticModel, "value-checkpoint.pt");
      torch::save(*criticModel_optimizer, "value-optimizer-checkpoint.pt");
      std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
  }*/
  
  itCounter++;
  
  return fLoss;
}

PPO::~PPO()
{
  saveFile.close();
  delete device;
  delete actorModel_optimizer;
  delete criticModel_optimizer;
  delete buffer;
  //google::protobuf::ShutdownProtobufLibrary();
  //delete logger;
}

PPO* createPPO(int inStateSize, int inActionSize, float policy_lr, float value_lr, float inGamma, int inHorizon, int epoch, int inMiniBatchSize, float inEps, float inLambda)
{
  return new PPO(inStateSize, inActionSize, policy_lr, value_lr, inGamma, inHorizon, epoch, inMiniBatchSize, inEps, inLambda);
}

void freePPO(PPO* ppo)
{
  delete ppo;
}

int64_t chooseAction(PPO* ppo, float* state)
{
  int64_t result = ppo->chooseAction(state);
  return result;
}

float remember(PPO* ppo, float* state, int64_t action, float reward, int64_t done)
{
  float result = ppo->remember(state, action, reward, done);
  return result;
}

extern "C"
{
  typedef struct PPO PPO;
void* createAgentc(int inStateSize, int inActionSize, float policy_lr, float value_lr, float inGamma, int inHorizon, int epoch, int inMiniBatchSize, float inEps, float inLambda)
  {
    return (void*)createPPO(inStateSize, inActionSize, policy_lr, value_lr, inGamma, inHorizon, epoch, inMiniBatchSize, inEps, inLambda);
  }
  
  void freeAgentc(void* ppo)
  {
    freePPO((PPO*)ppo);
  }
  
  int64_t chooseActionc(void* ppo, float* state)
  {
    return chooseAction((PPO*)ppo, state);
  }
  
  float rememberc(void* ppo, float* state, int64_t action, float reward, int64_t done)
  {
    return remember((PPO*)ppo, state, action, reward, done);
  }
}
