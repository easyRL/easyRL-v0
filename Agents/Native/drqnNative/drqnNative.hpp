#ifndef DRQNNATIVE_HPP
#define DRQNNATIVE_HPP
#include <torch/torch.h>
#include "drqnReplayBuffer.hpp"

using namespace torch;
using namespace std;

struct DuelingImpl: nn::Module
{
  DuelingImpl(int stateSize, int outputSize, int layerSize, int historySize) : 
    rnn(nn::RNNOptions(stateSize, layerSize).num_layers(2).batch_first(true)),
    qrnn(nn::RNNOptions(layerSize, layerSize).num_layers(2).batch_first(true)),
    aaqrnn(nn::RNNOptions(layerSize, layerSize).num_layers(2).batch_first(true)),
    qlin(nn::LinearOptions(layerSize*historySize, outputSize)),
    aaqlin(nn::LinearOptions(layerSize*historySize, outputSize))
  {
    register_module("rnn", rnn);
    register_module("qrnn", qrnn);
    register_module("aaqrnn", aaqrnn);
    register_module("qlin", qlin);
    register_module("aaqlin", aaqlin);
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor mask) {
    auto xtup = rnn(x);
    
    auto qtup = qrnn(get<0>(xtup), get<1>(xtup));
    Tensor q = get<0>(qtup);
    q = q.reshape({q.sizes()[0], q.sizes()[1] * q.sizes()[2]});
    q = qlin(q);
    
    auto aaqtup = aaqrnn(get<0>(xtup), get<1>(xtup));
    Tensor aaq = get<0>(aaqtup);
    aaq = aaq.reshape({aaq.sizes()[0], aaq.sizes()[1] * aaq.sizes()[2]});
    aaq = aaqlin(aaq);

    return mask * ((q + aaq) - aaq.mean(1, true));
  }

  nn::RNN rnn, qrnn, aaqrnn;
  nn::Linear qlin, aaqlin;
};

TORCH_MODULE(Dueling);

class DRQN
{
    public:
        DRQN(int inStateSize, int inActionSize, float inGamma, int inBatchSize, int inMemorySize, int inTargetUpdate, int inHistorySize);
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
        
        int stateSize;
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
    __declspec(dllexport) void* createAgentc(int stateSize, int actionSize, float gamma, int inBatchSize, int inMemorySize, int inTargetUpdate, int inHistorySize);
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
