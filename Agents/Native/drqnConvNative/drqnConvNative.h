void* createAgentc(int stateChannels, int stateDim1, int stateDim2, int actionSize, float gamma, int inBatchSize, int inMemorySize, int inTargetUpdate, int historySize, float learningRate);
void freeAgentc(void* agent);
int64_t chooseActionc(void* dqn, float* state);
float rememberc(void* dqn, float* state, int64_t action, float reward, int64_t done, int isTrain);
void savec(void* dqn, char* filename);
void loadc(void* dqn, char* filename);
void* memsavec(void* dqn);
void memloadc(void* dqn, void* mem);
