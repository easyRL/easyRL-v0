initial learning rate = 0.2
gamma = 0.7
epsilon decay = not mentioned

# For advanced feature, add following codes to model.py, after line 40
if self.environment.displayName == 'Custom_DrugDosing':
	self.environment.env.set_agent(self.agent)
	
# For advanced feature, add following codes to model.py, after line 68
if self.environment.displayName == 'Custom_DrugDosing':
	self.environment.env.episode_finish(episode)
	epsilon = min_epsilon + (max_epsilon - min_epsilon) * ((total_episodes - episode) / total_episodes)