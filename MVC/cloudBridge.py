import boto3

class CloudBridge:
    def __init__(self, jobID, secretKey, accessKey, name):
        self.animationFrames = []
        self.name = name
        self.jobID = jobID
        self.secretKey = secretKey
        self.accessKey = accessKey

        # Step Variables
        self.episodeAccEpsilon = 0
        self.episodeAccReward = 0
        self.episodeAccLoss = 0
        self.curEpisodeSteps = 0

        # Episode Variables
        self.trainingEpisodes = 0

    def refresh(self):
        self.animationFrames.clear()

    def submitStep(self, frame, epsilon, reward, loss):
        self.animationFrames.append(frame)

        # Accumulate Step
        self.episodeAccEpsilon += epsilon
        self.episodeAccReward += reward
        self.episodeAccLoss += loss
        self.curEpisodeSteps += 1

    def submitEpisode(self, episode):
        self.trainingEpisodes += 1

        # Redraw Graph
        avgLoss = self.episodeAccLoss / self.curEpisodeSteps
        totalReward = self.episodeAccReward
        avgEpsilon = self.episodeAccEpsilon / self.curEpisodeSteps

        graphPoints = (avgLoss, totalReward, avgEpsilon)

        self.curEpisodeSteps = 0
        self.episodeAccLoss = 0
        self.episodeAccReward = 0
        self.episodeAccEpsilon = 0

        if (len(self.animationFrames) > 0):
            self.animationFrames[0].save('./' + self.name + '-episode-' + str(episode) + ".gif", save_all=True, append_images=self.animationFrames)
        
    def submitTrainFinish(self):
        totalReward = self.episodeAccReward
        avgReward = totalReward / self.trainingEpisodes

        self.refresh()
