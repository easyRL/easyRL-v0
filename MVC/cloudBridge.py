import boto3
import uuid

class CloudBridge:
    def __init__(self, jobID, secretKey, accessKey, sessionToken):
        self.animationFrames = []
        self.jobID = jobID
        self.secretKey = secretKey
        self.accessKey = accessKey
        self.s3Client = None

        self.botoSession = boto3.Session (
            aws_access_key_id = accessKey,
            aws_secret_access_key = secretKey,
            aws_session_token = sessionToken, 
            region_name = 'us-east-1'
        )

        # Episode Variables
        self.trainingEpisodes = 0

        # If JobID is null generate one (WebGUI will pass in JobID)
        if (self.jobID is None):
            self.jobID = uuid.uuid4()
            
        self.refresh()
        self.init()

    # Create bucket for job in S3 to store data.
    def init(self):
        if self.s3Client is None:
            self.s3Client = self.botoSession.resource('s3')
            bucketName = 'easyrl-' + str(self.jobID)
            print(bucketName)
            self.s3Client.create_bucket(Bucket=bucketName)
            print("Created bucket for job.")
        pass

    # Kill infrastructure
    def terminate(self):
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

        # We may not want to upload gifs for all episodes.
        # May make it timed based in the event that lots of episodes are going quickly
        if (len(self.animationFrames) > 0):
            filename = self.state + '-episode-' + str(episode) + ".gif"
            self.animationFrames[0].save("./" + filename, save_all=True, append_images=self.animationFrames)
            if self.s3Client is None:
                s3Client.upload_file(filename, 'easyrl-' + str(self.jobID), filename)
        
    def submitTrainFinish(self):
        totalReward = self.episodeAccReward
        avgReward = totalReward / self.trainingEpisodes

        self.state = "Finished"
