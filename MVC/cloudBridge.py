import boto3
import uuid
import json
import time
import os
import math

class CloudBridge:

    def __init__(self, jobID, secretKey, accessKey, sessionToken, model):
        self.animationFrames = []
        self.jobID = jobID
        self.secretKey = secretKey
        self.accessKey = accessKey
        self.s3Client = None
        self.episodeData = []
        self.gifURLs = []
        self.delayTime = 1000
        self.uploadModels = True
        self.model = model

        self.lastSave = 0

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
            self.s3Client = self.botoSession.client('s3')
            bucketName = 'easyrl-' + str(self.jobID)
            print(bucketName)
            self.s3Client.create_bucket(Bucket=bucketName)
            print("Created bucket for job.")

    # Kill infrastructure
    def terminate(self):
        pass

    def upload(self, filename):
        if self.s3Client is None:
            self.s3Client.upload_file(filename, 'easyrl-' + str(self.jobID), filename)

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

        # Append data to data structure
        # e = episdoe, l = averageloss, r = totalreward, p = avgEpsilon
        self.episodeData.append({
            "e": episode, 
            "l": round(avgLoss,3),
            "p": round(avgEpsilon,3),
            "r": round(totalReward,3)
        })

        self.curEpisodeSteps = 0
        self.episodeAccLoss = 0
        self.episodeAccReward = 0
        self.episodeAccEpsilon = 0

        if (len(self.episodeData)) > 1000:
            self.episodeData.pop()

        currentTime = int(round(time.time() * 1000))
        if (currentTime - self.lastSave) > self.delayTime:
            self.lastSave = currentTime

            if (self.state == "Training" and self.uploadModels):
                self.model.save("continuousTraining.bin")

            # Render Gif
            if (len(self.animationFrames) > 0):
                filename = self.state + '-episode-' + str(episode) + ".gif"
                self.animationFrames[0].save("./" + filename, save_all=True, append_images=self.animationFrames)
                self.s3Client.upload_file(filename, 'easyrl-' + str(self.jobID), filename, ExtraArgs={'ACL': 'public-read'})
                os.remove("./" + filename)
                self.gifURLs.append("https://easyrl-" + str(self.jobID) + ".s3.amazonaws.com/" + filename)

                if (len(self.gifURLs)) > 10:
                    self.gifURLs.pop()

            payload =  {
                "totalReward": self.episodeAccReward,
                "avgReward": self.episodeAccReward / self.trainingEpisodes,
                "episodes": self.episodeData,
                "gifs": self.gifURLs
            }

            with open('data.json', 'w+') as f:
                json.dump(payload, f)                

        self.animationFrames = []
        
    def submitTrainFinish(self):
        totalReward = self.episodeAccReward
        avgReward = self.episodeAccReward / self.trainingEpisodes

        self.state = "Finished"
