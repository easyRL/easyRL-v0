typedef struct DQN DQN;

DQN* createDQNc(int stateSize, int actionSize, float gamma, int inBatchSize, int inMemorySize, int inTargetUpdate);
void freeDQNc(DQN* dqn);
int64_t chooseActionc(DQN* dqn, float* state);
float rememberc(DQN* dqn, float* state, int64_t action, float reward, int64_t done);
void savec(DQN* dqn, char* filename);
void loadc(DQN* dqn, char* filename);
void* memsavec(DQN* dqn);
void memloadc(DQN* dqn, void* mem);
