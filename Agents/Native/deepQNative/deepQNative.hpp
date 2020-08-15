#ifndef DQN_H
#define DQN_H
#include <torch/torch.h>
#include "dqnReplayBuffer.hpp"

using namespace torch;

struct DuelingImpl: nn::Module
{
  DuelingImpl(int stateSize, int outputSize, int layerSize) : 
    lin1(nn::LinearOptions(stateSize, layerSize)),
    lin2(nn::LinearOptions(layerSize, layerSize)),
    qlin1(nn::LinearOptions(layerSize, layerSize)),
    qlin2(nn::LinearOptions(layerSize, outputSize)),
    aaqlin1(nn::LinearOptions(layerSize, layerSize)),
    aaqlin2(nn::LinearOptions(layerSize, outputSize))
  {
    register_module("lin1", lin1);
    register_module("lin2", lin2);
    register_module("qlin1", qlin1);
    register_module("qlin2", qlin2);
    register_module("aaqlin1", aaqlin1);
    register_module("aaqlin2", aaqlin2);
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor mask) {
    x = torch::relu(lin1(x)); 
    x = torch::relu(lin2(x));
    Tensor q = torch::relu(qlin1(x));
    q = qlin2(q);
    Tensor aaq = torch::relu(aaqlin1(x));
    aaq = aaqlin2(aaq);

    return mask * ((q + aaq) - aaq.mean(1, true));
  }

  nn::Linear lin1, lin2, qlin1, qlin2, aaqlin1, aaqlin2;
};

TORCH_MODULE(Dueling);

class DQN
{
    public:
        DQN(int stateSize, int actionSize, float gamma, int inBatchSize, int inMemorySize, int inTargetUpdate, float learningRate);
        ~DQN();
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
        int64_t itCounter;
        int64_t checkpoint_counter;
};

#ifdef _WIN32
extern "C"
{
    __declspec(dllexport) void* createAgentc(int stateSize, int actionSize, float gamma, int inBatchSize, int inMemorySize, int inTargetUpdate, float learningRate);
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
