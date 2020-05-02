class ExperienceBuffer:
    def __init__(self):
        self.buffer = []

    def add_transition(self, transition):
        self.buffer.append(transition)

    def sample_randomly(self, batch_size):
        pass