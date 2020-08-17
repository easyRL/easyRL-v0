#ifndef DRQNCONVNATIVE_HPP
#define DRQNCONVNATIVE_HPP
#include <torch/torch.h>
#include "drqnConvReplayBuffer.hpp"
//#include <tensorboard_logger.h>

using namespace torch;
using namespace torch::indexing;
using namespace std;

struct DuelingImpl: nn::Module
{
  DuelingImpl(int stateChannels, int stateDim1, int stateDim2, int outputSize, int layerSize, int historySize, int kerncount1, int kerncount2, int kerncount3, int kernsize1, int kernsize2, int kernsize3, int stride1, int stride2, int stride3)
  {
    conv1outx = ((stateDim1-kernsize1+1)-1)/stride1+1;
    conv1outy = ((stateDim2-kernsize1+1)-1)/stride1+1;
  
    conv2outx = ((conv1outx-kernsize2+1)-1)/stride2+1;
    conv2outy = ((conv1outy-kernsize2+1)-1)/stride2+1;
  
    conv3outx = ((conv2outx-kernsize3+1)-1)/stride3+1;
    conv3outy = ((conv2outx-kernsize3+1)-1)/stride3+1;
  
    conv1 = nn::Conv2d(nn::Conv2dOptions(stateChannels,kerncount1,kernsize1).stride(stride1));
    //nn::init::kaiming_normal_(conv1->weight, 0, torch::kFanIn, torch::kReLU);
    //nn::init::zeros_(conv1->bias);
    
    conv2 = nn::Conv2d(nn::Conv2dOptions(kerncount1,kerncount2,kernsize2).stride(stride2));
    //nn::init::kaiming_normal_(conv2->weight, 0, torch::kFanIn, torch::kReLU);
    //nn::init::zeros_(conv2->bias);
    
    conv3 = nn::Conv2d(nn::Conv2dOptions(kerncount2,kerncount3,kernsize3).stride(stride3));
    
    //qrnn = nn::RNN(nn::RNNOptions(conv2outx*conv2outy*kerncount2, layerSize).num_layers(2).batch_first(true));
    
    //aaqrnn = nn::RNN(nn::RNNOptions(conv2outx*conv2outy*kerncount2, layerSize).num_layers(1).batch_first(true));
    
    qlin1 = nn::Linear(nn::LinearOptions(historySize*conv3outx*conv3outy*kerncount3, layerSize));
    
    qlin2 = nn::Linear(nn::LinearOptions(layerSize, outputSize));
    //nn::init::kaiming_normal_(qlin->weight, 0, torch::kFanIn, torch::kReLU);
    //nn::init::zeros_(qlin->bias);
    
    //aaqlin = nn::Linear(nn::LinearOptions(layerSize*historySize, outputSize));
    //nn::init::kaiming_normal_(aaqlin->weight, 0, torch::kFanIn, torch::kReLU);
    //nn::init::zeros_(aaqlin->bias);

    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    //register_module("qrnn", qrnn);
    //register_module("aaqrnn", aaqrnn);
    register_module("qlin1", qlin1);
    register_module("qlin2", qlin2);
    //register_module("aaqlin", aaqlin);
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor mask) {
    x = x/128.0 - 1.0;
    
    Tensor xView = x.view({x.sizes()[0]*x.sizes()[1], x.sizes()[2], x.sizes()[3], x.sizes()[4]});
    
    xView = torch::relu(conv1(xView));
    xView = torch::relu(conv2(xView));
    xView = torch::relu(conv3(xView));
    
    xView = xView.view({x.sizes()[0], x.sizes()[1]*xView.sizes()[1]*xView.sizes()[2]*xView.sizes()[3]});
    
    //auto qtup = qrnn(xView);
    //Tensor q = torch::relu(get<0>(qtup));
    //q = q.reshape({q.sizes()[0], q.sizes()[1] * q.sizes()[2]});
    
    Tensor q = torch::relu(qlin1(xView));
    q = qlin2(q);
    
    /*auto aaqtup = aaqrnn(xView);
    Tensor aaq = leaky(get<0>(aaqtup));
    aaq = aaq.reshape({aaq.sizes()[0], aaq.sizes()[1] * aaq.sizes()[2]});
    aaq = aaqlin(aaq);*/

    return mask * q;
    //return mask * ((q + aaq) - aaq.mean(1, true));
  }

  nn::Conv2d conv1 = nullptr, conv2 = nullptr, conv3 = nullptr;
  //nn::RNN qrnn = nullptr;, aaqrnn = nullptr;
  nn::Linear qlin1 = nullptr, qlin2 = nullptr;//, aaqlin = nullptr;
  
  int conv1outx, conv1outy, conv2outx, conv2outy, conv3outx, conv3outy;
};

TORCH_MODULE(Dueling);

class DRQN
{
    public:
        DRQN(int inStateChannels, int inStateDim1, int inStateDim2, int inActionSize, float inGamma, int inBatchSize, int inMemorySize, int inTargetUpdate, int inHistorySize, float inLearningRate);
        ~DRQN();
        int64_t chooseAction(float* state);
        float remember(float* state, int64_t action, float reward, int64_t done, int isTrain);
        void newEpisode(){}
        void save(char* filename);
        void load(char* filename);
        std::stringstream* memsave();
        void memload(std::stringstream* mem);
        
    private:
        torch::Device* device;
        Dueling model = nullptr;
        Dueling target = nullptr;
        torch::optim::Optimizer* model_optimizer;
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
        int skipCounter;
        int64_t checkpoint_counter;
        bool choiceFlag;
        
        
        //TensorBoardLogger* logger = nullptr;
};

#ifdef _WIN32
extern "C"
{
    __declspec(dllexport) void* createAgentc(int stateChannels, int stateDim1, int stateDim2, int actionSize, float gamma, int inBatchSize, int inMemorySize, int inTargetUpdate, int historySize, float inLearningRate);
    __declspec(dllexport) void freeAgentc(void* dqn);
    __declspec(dllexport) int64_t chooseActionc(void* dqn, float* state);
    __declspec(dllexport) float rememberc(void* dqn, float* state, int64_t action, float reward, int64_t done, int isTrain);
    __declspec(dllexport) void savec(void* dqn, char* filename);
    __declspec(dllexport) void loadc(void* dqn, char* filename);
    __declspec(dllexport) void* memsavec(void* dqn);
    __declspec(dllexport) void memloadc(void* dqn, void* mem);
}
#endif
#endif
