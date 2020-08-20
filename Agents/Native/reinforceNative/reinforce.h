void* createAgentc(int inStateSize, int inActionSize, float policy_lr, float inGamma);
void freeAgentc(void* agent);
int64_t chooseActionc(void* actorCritic, float* state);
float rememberc(void* actorCritic, float* state, int64_t action, float reward, int64_t done);
