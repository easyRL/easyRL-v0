#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "drqnConvNative.hpp"

using namespace std;

const int layerSize = 256;

DRQN::DRQN(int inStateChannels, int inStateDim1, int inStateDim2, int inActionSize, float inGamma, int inBatchSize, int inMemorySize, int inTargetUpdate, int inHistorySize, float inLearningRate)
{
  //GOOGLE_PROTOBUF_VERIFY_VERSION;

  stateChannels = inStateChannels;
  stateDim1 = inStateDim1;
  stateDim2 = inStateDim2;
  actionSize = inActionSize;
  gamma = inGamma;
  batchSize = inBatchSize;
  memorySize = inMemorySize;
  targetUpdate = inTargetUpdate;
  historySize = inHistorySize;
  
  //remove("logs/tfevents.pb");
  //logger = new TensorBoardLogger("logs/tfevents.pb");

  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = new torch::Device(torch::kCUDA);
  }
  else
  {
    device = new torch::Device(torch::kCPU);
  }
  
  model = Dueling(stateChannels, stateDim1, stateDim2, actionSize, layerSize, historySize, 16, 32, 32, 8, 4, 3, 4, 2, 1);
  model->to(*device);
  target = Dueling(stateChannels, stateDim1, stateDim2, actionSize, layerSize, historySize, 16, 32, 32, 8, 4, 3, 4, 2, 1);
  target->to(*device);
  
  model_optimizer = new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(inLearningRate).eps(0.001));
  
  fullMask = torch::ones({1,actionSize}).to(*device);
  
  replay = new ReplayBuffer(stateChannels*stateDim1*stateDim2, memorySize, batchSize, historySize);
  printf("BUFFERSIZE: %lu\n", sizeof(float)*memorySize*stateChannels*stateDim1*stateDim2);
  
  itCounter = 0;
  checkpoint_counter = 0;

  cout << "learning rate: " << inLearningRate << endl << "target update rate: " << targetUpdate << endl;
  cout << "stateSize: (" << stateChannels << ", " << stateDim1 << ", " << stateDim2 << "), actionSize: " << actionSize << ", gamma: " << gamma << endl;
  cout << ", batchSize: " << batchSize << ", memorySize: " << memorySize << ", targetUpdate: " << targetUpdate << endl;
  cout << "historySize: " << historySize << endl;
}

int64_t DRQN::chooseAction(float* state)
{
  //cout << "chooseAction" << endl;
  
  if (replay->curSize == 0)
  {
    return 0;
  }
  
  int64_t action;
  float* recent = new float[historySize * stateChannels * stateDim1 * stateDim2];
  replay->recent(recent, state);
  
  Tensor xsingle = torch::from_blob(recent, {1, historySize, stateChannels, stateDim1, stateDim2}).to(*device);
  Tensor ysingle = model->forward(xsingle, fullMask);
  
  /*cout << "XSINGLE:\n" << xsingle << "\n\n\n";
  int dummy;
  cin >> dummy;*/
  
  action = ysingle.argmax(1).item().toInt();
  
  //cout << "XSINGLE:\n" << xsingle << "\n\n\n";
  
    //action = rand()%outputSize;
    //std::cout << "ACTION " << action << " RANDOMED" << std::endl;
  delete[] recent;
  return action;
}

float DRQN::remember(float* state, int64_t action, float reward, int64_t done, int isTrain)
{
  //cout << "REMEMBERED!\n";
  //cout << "remember" << endl;
  float fLoss=0;
  model_optimizer->zero_grad();
  replay->add(state, action, reward, done);
  
  if (isTrain && replay->curSize >= batchSize)
  {
    float* bStates = new float[batchSize * historySize * stateChannels * stateDim1 * stateDim2];
    int64_t* bActions = new int64_t[batchSize];
    float* bRewards = new float[batchSize];
    float* bNextStates = new float[batchSize * historySize * stateChannels * stateDim1 * stateDim2];
    int64_t* bDones = new int64_t[batchSize];
    
    replay->sample((float*)bStates, (int64_t*)bActions, (float*)bRewards, (float*)bNextStates, (int64_t*)bDones);
    
    Tensor xbatch = torch::from_blob(bStates, {batchSize, historySize, stateChannels, stateDim1, stateDim2}).to(*device);        
    Tensor actionsbatch = torch::from_blob(bActions, {batchSize, 1}, TensorOptions().dtype(kInt64)).to(*device);
    Tensor rewardsbatch = torch::from_blob(bRewards, {batchSize, 1}).to(*device);
    Tensor nextxbatch = torch::from_blob(bNextStates, {batchSize, historySize, stateChannels, stateDim1, stateDim2}).to(*device);
    Tensor donesbatch = torch::from_blob(bDones, {batchSize, 1}, TensorOptions().dtype(kInt64)).to(*device);
    
    /*cout << "XBATCH:\n" << xbatch << "\n\n\n";
    int dummy;
    cin >> dummy;*/
    
    
    Tensor actionsOneHotBatch = (torch::zeros({batchSize, actionSize}).to(*device).scatter_(1, actionsbatch, 1)).to(*device);
    Tensor ybatch = model->forward(xbatch, actionsOneHotBatch);
    Tensor nextybatch = model->forward(nextxbatch, fullMask);
    Tensor nextybatchTarg = target->forward(nextxbatch, fullMask);
    Tensor argmaxes = nextybatch.argmax(1, true);
    Tensor maxes = nextybatchTarg.gather(1, argmaxes);
    Tensor nextvals = rewardsbatch + (1 - donesbatch) * (gamma * maxes);    
    
    Tensor targetbatch = torch::zeros({batchSize, actionSize}).to(*device).scatter_(1, actionsbatch, nextvals).detach();    
    
    torch::Tensor loss = torch::smooth_l1_loss(ybatch, targetbatch.detach());
    loss.backward();
    
    //nn::utils::clip_grad_value_(model->parameters(), 1.0);
    
    model_optimizer->step();
    fLoss = loss.item<float>();
    
    
    if ((itCounter+1) % targetUpdate == 0)
    {
      std::stringstream stream;
      torch::save(model, stream);
      torch::load(target, stream);
      std::cout << "target updated" << std::endl;
      
      /*auto items = model->named_parameters().items();
      for (auto item : items)
      {
        auto key = item.key();
        auto val = item.value();
        auto gradVal = val.grad();
        int len = 1, gradLen = 1;
        for (int i=0; i<val.sizes().size(); i++)
        {
          len *= val.sizes()[i];
        }
        for (int i=0; i<gradVal.sizes().size(); i++)
        {
          gradLen *= gradVal.sizes()[i];
        }
        const int clen = len, gradClen = gradLen;
        val = val.view({clen}).to(torch::kCPU);
        gradVal = gradVal.view({gradClen}).to(torch::kCPU);
        assert(val.is_contiguous());
        assert(gradVal.is_contiguous());
        auto weight_a = val.accessor<float,1>();
        auto grad_a = gradVal.accessor<float,1>();
        //cout << key << " added\n\n\n";
        logger->add_histogram(key, itCounter, &weight_a[0], len);
        logger->add_histogram(key+".grad", itCounter, &grad_a[0], gradLen);
        if (key == "conv1.weight")
        {
          int ind = rand()%gradLen;
          cout << "Sample grad: " << grad_a[ind] << endl;
        }
      }*/
      
      //cout << "\n\n\n\n\n\n\nXBATCH:\n" << xbatch << endl;
      
      /*cout << "\n\n\n\n\n*************************************\n";
      cout << xbatch << endl;
      cout << "*************************************\n\n\n\n\n";*/
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

void DRQN::save(char* filename)
{
  torch::save(model, filename);
}

void DRQN::load(char* filename)
{
  torch::load(model, filename);
  cout << "loaded " << filename << endl;
}

std::stringstream* DRQN::memsave()
{
  std::stringstream* mem = new std::stringstream;
  torch::save(model, *mem);
  return mem;
}

void DRQN::memload(std::stringstream* mem)
{
  torch::load(model, *mem);
  delete mem;
}

DRQN::~DRQN()
{
  delete device;
  delete model_optimizer;
  delete replay;
  //delete logger;
  //google::protobuf::ShutdownProtobufLibrary();
}

DRQN* createDRQN(int stateChannels, int stateDim1, int stateDim2, int actionSize, float gamma, int inBatchSize, int inMemorySize, int inTargetUpdate, int inHistorySize, float inLearningRate)
{
  return new DRQN(stateChannels, stateDim1, stateDim2, actionSize, gamma, inBatchSize, inMemorySize, inTargetUpdate, inHistorySize, inLearningRate);
}

void freeDRQN(DRQN* drqn)
{
  delete drqn;
}

int64_t chooseAction(DRQN* drqn, float* state)
{
  int64_t result = drqn->chooseAction(state);
  return result;
}

float remember(DRQN* drqn, float* state, int64_t action, float reward, int64_t done, int isTrain)
{
  float result = drqn->remember(state, action, reward, done, isTrain);
  return result;
}

void save(DRQN* drqn, char* filename)
{
  drqn->save(filename);
}

void load(DRQN* drqn, char* filename)
{
  drqn->load(filename);
}

void* memsave(DRQN* drqn)
{
  return (void*)drqn->memsave();
}

void memload(DRQN* drqn, void* mem)
{
  drqn->memload((std::stringstream*)mem);
}

extern "C"
{
  typedef struct DRQN DRQN;
void* createAgentc(int stateChannels, int stateDim1, int stateDim2, int actionSize, float gamma, int inBatchSize, int inMemorySize, int inTargetUpdate, int inHistorySize, float inLearningRate)
  {
    return (void*)createDRQN(stateChannels, stateDim1, stateDim2, actionSize, gamma, inBatchSize, inMemorySize, inTargetUpdate, inHistorySize, inLearningRate);
  }
  
  void freeAgentc(void* drqn)
  {
    freeDRQN((DRQN*)drqn);
  }
  
  int64_t chooseActionc(void* drqn, float* state)
  {
    return chooseAction((DRQN*)drqn, state);
  }
  
  float rememberc(void* drqn, float* state, int64_t action, float reward, int64_t done, int isTrain)
  {
    return remember((DRQN*)drqn, state, action, reward, done, isTrain);
  }
  
  void savec(void* drqn, char* filename)
  {
    save((DRQN*)drqn, filename);
  }
  
  void loadc(void* drqn, char* filename)
  {
    load((DRQN*)drqn, filename);
  }
  
  void* memsavec(void* drqn)
  {
    return memsave((DRQN*)drqn);
  }
  
  void memloadc(void* drqn, void* mem)
  {
    memload((DRQN*)drqn, mem);
  }
}
