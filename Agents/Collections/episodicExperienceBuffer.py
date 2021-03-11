import random


class EpisodicExperienceBuffer:
    def __init__(self, stored_tuple_length: int, max_length: int, default_tuple: tuple):
        assert len(default_tuple) == stored_tuple_length, "Default tuple should be of size " + str(stored_tuple_length)

        self.in_size = stored_tuple_length
        self.max_transitions = max_length
        self.default_tuple = default_tuple
        self.size = 0
        # Explanation of internal storage:
        # self.episodes is a list of episodes. Each episode is a tuple of lists, where the number of lists is determined by the user
        # The lists contain all the transitions for that episode, separated from each other
        # Because of this, you will see a lot of [0]s. These are getting the first element of the tuple, usually to get the first list
        # to determine its length.
        self.episodes = [self.__create_empty_lists_tuple(self.in_size)]

    # Pass in your transitions in the form: add_transition(a, b, c, d)
    # If you wish to truncate your episode with this transition, add ', truncate_episode=False' as the last argument
    def add_transition(self, *transition, truncate_episode=False):
        assert isinstance(transition, tuple) and len(transition) == self.in_size, "Transition added should be of size " + str(self.in_size)
        self.__mass_append(self.episodes[-1], *transition)
        self.size += 1
        if truncate_episode:
            self.truncate_episode()
        self.__check_size()

    # Give batch size and history length. Method will find a random episode of at least batch_size transitions length.
    # If none that large exist, a random episode will be chosen.
    # For each sample in the batch, a random index from the same episode will be chosen. (If episode was of sufficient
    # length then these indexes will not be repeated)
    # history_length number of sequential transitions are then found starting at the randomly chosen indexes and working backwards
    # (but without reversing the order of the transitions).
    # Output is of the form: tuple(batch_list(history_list)), i.e. tuple_size x batch_size x history_length
    # When calling this method, you should assign the returned tuple to multiple corresponding variables
    # Example: observations, actions, rewards, dones, infos = sample_randomly_in_episode(#, #). Batches will match across variables
    def sample_randomly_in_episode(self, batch_size: int, history_length: int) -> tuple:
        assert len(self.episodes) > 1 or len(self.episodes[0][0]) > 0, "Cannot sample from an empty buffer"
        substantial_episodes = [episode for episode in self.episodes if len(episode[0]) >= batch_size]
        if len(substantial_episodes) < 1:
            substantial_episodes = [episode for episode in self.episodes if len(episode[0]) > 0]
            episode_to_sample = random.sample(substantial_episodes, 1)[0]
            indexes = random.choices([x for x in range(len(episode_to_sample[0]))], k=batch_size)
        else:
            episode_to_sample = random.sample(substantial_episodes, 1)[0]
            indexes = random.sample([x for x in range(len(episode_to_sample[0]))], batch_size)
        return self.__get_batch(episode_to_sample, indexes, history_length)

    # Returns a list for each element in the episode tuple, of length history_length
    def __get_batch(self, episode: tuple, indexes_in_episode: list, history_length: int) -> tuple:
        output = self.__create_empty_lists_tuple(self.in_size)
        for tuple_list_index in range(len(output)):
            current_batch = output[tuple_list_index]
            for start_index in indexes_in_episode:
                current_sample = []
                for index in range(start_index - history_length + 1, start_index + 1):
                    if index < 0:
                        current_sample.append(self.default_tuple[tuple_list_index])
                    else:
                        current_sample.append(episode[tuple_list_index][index])
                current_batch.append(current_sample)
        return output

    # Gets most recent transitions, history_length timesteps back. returns tuple_length x history_length, i.e. a tuple of lists of length history_length
    def get_recent_transition(self, history_length: int) -> tuple:
        recent_episode = self.episodes[-1]
        episode_length = len(recent_episode[0])
        # Gets recent transition as batch, then removes the unnecessary dimension of length 1
        return tuple([tuple_list[0] for tuple_list in self.__get_batch(recent_episode, [episode_length - 1], history_length)])

    def __check_size(self):
        if self.size > self.max_transitions:  # If too large
            self.size -= self.episodes[0][0]  # Reduce size by number of transitions deleted
            del self.episodes[0]  # Delete oldest episode

    # Marks the ends the current episode in the buffer's memory
    def truncate_episode(self):
        self.episodes.append(self.__create_empty_lists_tuple(self.in_size))

    @staticmethod
    def __create_empty_lists_tuple(tuple_length: int) -> tuple:
        output = tuple()
        for x in range(tuple_length):
            output = output + ([],)
        return output

    @staticmethod
    def __mass_append(lists: tuple, *args):
        for index in range(min(len(lists), len(args))):
            lists[index].append(args[index])

    def __len__(self):
        return self.size

# TESTING:
# a = EpisodicExperienceBuffer(3, 40, (0, 0, 0))
# a.add_transition(1, 1, 1)
# a.add_transition(5, 5, 5, truncate_episode=True)
# a.add_transition(3, 3, 3)
# a.add_transition(4, 4, 4)
# a.add_transition(5, 5, 5, truncate_episode=True)
# a.add_transition(1, 1, 1)
# a.add_transition(2, 2, 2)
# a.add_transition(3, 3, 3)
# a.add_transition(4, 4, 4)
# a.add_transition(5, 5, 5, truncate_episode=True)
# a.add_transition(4, 4, 4)
# a.add_transition(4, 4, 4)
# a.add_transition(5, 5, 5)
# print(a.sample_randomly_in_episode(6, 3))
# print(a.sample_randomly_in_episode(3, 3))
# print(a.get_recent_transition(5))
# print(a.get_recent_transition(4))
# print(a.get_recent_transition(3))
# print(a.get_recent_transition(2))
# print(a.get_recent_transition(1))
# print(a.get_recent_transition(0))