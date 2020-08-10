void* createAgentc(int inStateSize, int inActionSize, float policy_lr, float value_lr, float inGamma, int inHorizon, int epoch, int inMiniBatchSize, float inEps, float inLambda);
void freeAgentc(void* agent);
int64_t chooseActionc(void* ppo, float* state);
float rememberc(void* ppo, float* state, int64_t action, float reward, int64_t done);
