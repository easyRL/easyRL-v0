#ifndef DRQNNATIVE_HPP
#define DRQNNATIVE_HPP
#include <torch/torch.h>
#include "drqnConvReplayBuffer.hpp"

using namespace torch;
using namespace torch::indexing;
using namespace std;

struct DuelingImpl: nn::Module
{
  DuelingImpl(int stateChannels, int stateDim1, int stateDim2, int outputSize, int layerSize, int historySize, int kerncount1, int kerncount2, int kernsize1, int kernsize2, int stride1, int stride2)
  {
    conv1outx = ((stateDim1-kernsize1+1)-1)/stride1+1;
    conv1outy = (int)(((stateDim2-kernsize1+1)-1)/stride1+1);
  
    conv2outx = ((conv1outx-kernsize2+1)-1)/stride2+1;
    conv2outy = ((conv1outy-kernsize2+1)-1)/stride2+1;
  
    conv1 = nn::Conv2d(nn::Conv2dOptions(stateChannels,kerncount1,kernsize1).stride(stride1));
    conv2 = nn::Conv2d(nn::Conv2dOptions(kerncount1,kerncount2,kernsize2).stride(stride2));
    qrnn = nn::RNN(nn::RNNOptions(conv2outx*conv2outy*kerncount2, layerSize).num_layers(1).batch_first(true));
    aaqrnn = nn::RNN(nn::RNNOptions(conv2outx*conv2outy*kerncount2, layerSize).num_layers(1).batch_first(true));
    qlin = nn::Linear(nn::LinearOptions(layerSize*historySize, outputSize));
    aaqlin = nn::Linear(nn::LinearOptions(layerSize*historySize, outputSize));
  
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("qrnn", qrnn);
    register_module("aaqrnn", aaqrnn);
    register_module("qlin", qlin);
    register_module("aaqlin", aaqlin);
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor mask) {
    
    Tensor xView = x.view({x.sizes()[0]*x.sizes()[1], x.sizes()[2], x.sizes()[3], x.sizes()[4]});
    
    xView = conv1(xView);
    xView = conv2(xView);
    
    xView = xView.view({x.sizes()[0], x.sizes()[1], xView.sizes()[1]*xView.sizes()[2]*xView.sizes()[3]});
    
    auto qtup = qrnn(xView);
    Tensor q = get<0>(qtup);
    q = q.reshape({q.sizes()[0], q.sizes()[1] * q.sizes()[2]});
    q = qlin(q);
    
    auto aaqtup = aaqrnn(xView);
    Tensor aaq = get<0>(aaqtup);
    aaq = aaq.reshape({aaq.sizes()[0], aaq.sizes()[1] * aaq.sizes()[2]});
    aaq = aaqlin(aaq);

    return mask * ((q + aaq) - aaq.mean(1, true));
  }

  nn::Conv2d conv1 = nullptr, conv2 = nullptr;
  nn::RNN qrnn = nullptr, aaqrnn = nullptr;
  nn::Linear qlin = nullptr, aaqlin = nullptr;
  
  int conv1outx, conv1outy, conv2outx, conv2outy;
};

TORCH_MODULE(Dueling);

class DRQN
{
    public:
        DRQN(int inStateChannels, int inStateDim1, int inStateDim2, int inActionSize, float inGamma, int inBatchSize, int inMemorySize, int inTargetUpdate, int inHistorySize);
        ~DRQN();
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
        
        int stateChannels;
        int stateDim1;
        int stateDim2;
        int actionSize;
        float gamma;
        int batchSize;
        int memorySize;
        int targetUpdate;
        int historySize;
        int64_t itCounter;
        int64_t checkpoint_counter;
};

#ifdef _WIN32
extern "C"
{
    __declspec(dllexport) void* createAgentc(int stateChannels, int stateDim1, int stateDim2, int actionSize, float gamma, int inBatchSize, int inMemorySize, int inTargetUpdate, int historySize);
    __declspec(dllexport) void freeAgentc(void* dqn);
    __declspec(dllexport) int64_t chooseActionc(void* dqn, float* state);
    __declspec(dllexport) float rememberc(void* dqn, float* state, int64_t action, float reward, int64_t done);
    __declspec(dllexport) void savec(void* dqn, char* filename);
    __declspec(dllexport) void loadc(void* dqn, char* filename);
    __declspec(dllexport) void* memsavec(void* dqn);
    __declspec(dllexport) void memloadc(void* dqn, void* mem);
}
#endif
#endif
