import boto3
import uuid

class CloudBridge:
    def __init__(self, jobID, secretKey, accessKey, sessionToken):
        self.animationFrames = []
        self.jobID = jobID
        self.secretKey = secretKey
        self.accessKey = accessKey

        self.s3Client = boto3.Session (
            aws_access_key_id = accessKey,
            aws_secret_access_key = secretKey,
            aws_session_token = sessionToken
        )

        # Episode Variables
        self.trainingEpisodes = 0

        # If JobID is null generate one (WebGUI will pass in JobID)
        if (self.jobID is None):
            self.jobID = uuid.uuid4()
            
        self.refresh()

    # Create bucket for job in S3 to store data.
    def init():
        pass

    # Kill infrastructure
    def terminate():
        pass

    def setState(self, state):
        self.state = state

    def refresh(self):
        self.state = "Idle"

        # Step Variables
        self.episodeAccEpsilon = 0
        self.episodeAccReward = 0
        self.episodeAccLoss = 0
        self.curEpisodeSteps = 0

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

        self.state = "Finished"
