typedef struct DQN DQN;
const int64_t stateSize = 4;

DQN* createDQNc();
void freeDQNc(DQN* dqn);
int64_t chooseActionc(DQN* dqn, float state[stateSize]);
float rememberc(DQN* dqn, float state[stateSize], int64_t action, float reward, int64_t done);
