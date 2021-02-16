import boto3

class CloudBridge:
    def __init__(self, jobID, secretKey, accessKey, name):
        self.animationFrames = []
        self.name = name
        self.jobID = jobID
        self.secretKey = secretKey
        self.accessKey = accessKey

    def refresh(self):
        self.animationFrames.clear()

    def submitStep(self, frame, epsilon, reward, loss):
        self.animationFrames.append(frame)

    def submitEpisode(self, episode):
        if (len(self.animationFrames) > 0):
            self.animationFrames[0].save('./' + self.name + '-episode-' + str(episode) + ".gif", save_all=True, append_images=self.animationFrames)
