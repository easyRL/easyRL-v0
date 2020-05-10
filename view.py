import tkinter
from tkinter import ttk
from PIL import Image, ImageTk
from model import Model


class View:
    """
    :param master: the top level widget of Tk
    :type master: tkinter.Tk
    :param listener: the listener object that will handle user input
    :type listener: controller.ViewListener
    """
    def __init__(self, master, listener):
        self.StartWindow(master, listener)

    class Window:
        def __init__(self, master, listener):
            self.master = master
            self.listener = listener
            self.frame = tkinter.Frame(master)
            for i in range(10):
                self.frame.grid_columnconfigure(i, minsize=75)
                self.frame.grid_rowconfigure(i, minsize=50)

        def backButton(self):
            self.frame.destroy()

    class StartWindow(Window):
        def __init__(self, master, listener):
            super().__init__(master, listener)

            self.projectsButton = tkinter.Button(self.frame, text='Projects', fg='black')
            self.projectsButton.grid(row=2, column=2, rowspan=2, columnspan = 2, sticky='wens')

            self.examplesButton = tkinter.Button(self.frame, text='Examples', fg='black')
            self.examplesButton.grid(row=2, column=6, rowspan=2, columnspan = 2, sticky='wens')

            self.newButton = tkinter.Button(self.frame, text='New', fg='black', command=self.handleButton)
            self.newButton.grid(row=6, column=4, rowspan=2, columnspan = 2, sticky='wens')

            self.frame.grid(row=0, column=0)
            self.frame.lift()

        def handleButton(self):
            View.EnvironmentChooser(self.master, self.listener)

    class EnvironmentChooser(Window):
        def __init__(self, master, listener):
            super().__init__(master, listener)

            self.backButton = tkinter.Button(self.frame, text='back', fg='black', command=self.backButton)
            self.backButton.grid(row=0, column=0, sticky='wens')

            self.title = tkinter.Label(self.frame, text='Select an Environment:')
            self.title.grid(row=1, column=4, columnspan=2, sticky='wens')

            self.frozenLakeButton = tkinter.Button(self.frame, text='Frozen Lake', fg='black', command=self.chooseFrozenLake)
            self.frozenLakeButton.grid(row=2, column=4, columnspan=2, sticky='wens')

            self.cartPoleButton = tkinter.Button(self.frame, text='Cart Pole', fg='black', command=self.chooseCartPole)
            self.cartPoleButton.grid(row=4, column=4, columnspan=2, sticky='wens')

            self.customButton = tkinter.Button(self.frame, text='Custom Environment', fg='black', command=self.chooseCustom)
            self.customButton.grid(row=6, column=4, columnspan=2, sticky='wens')

            self.frame.grid(row=0, column=0)
            self.frame.lift()

        def chooseFrozenLake(self):
            self.listener.setFrozenLakeEnv()
            View.ProjectWindow(self.master, self.listener)
            self.frame.destroy()

        def chooseCartPole(self):
            self.listener.setCartPoleEnv()
            View.ProjectWindow(self.master, self.listener)
            self.frame.destroy()

        def chooseCustom(self):
            pass

    class ProjectWindow(Window):
        def __init__(self, master, listener):
            super().__init__(master, listener)

            self.backButton = tkinter.Button(self.frame, text='<- Back', fg='black', command=self.backButton)
            self.backButton.grid(row=0, column=0)

            self.tab = ttk.Notebook(self.frame)
            self.tab.bind("<<NotebookTabChanged>>", self.tabChange)

            self.qlearningTab = View.QLearningTab(self.tab, listener)
            self.deepQTab = tkinter.Frame(self.tab)
            self.deepSarsaTab = tkinter.Frame(self.tab)
            self.tab.add(self.qlearningTab, text='Q Learning')
            self.tab.add(self.deepQTab, text='Deep Q Learning')
            self.tab.add(self.deepSarsaTab, text='Etc.')

            self.tab.grid(row=1, column=0, rowspan=9, columnspan=10, sticky='wens')

            self.frame.grid(row=0, column=0)
            self.frame.lift()

        def tabChange(self, event):
            tabIndex = event.widget.index('current')
            if tabIndex == 0:
                self.listener.setQLearningAgent()
            elif tabIndex == 1:
                self.listener.setDeepQLearningAgent()
            elif tabIndex == 2:
                self.listener.setDeepSarsaAgent()

    class QLearningTab(tkinter.Frame):
        def __init__(self, tab, listener):
            super().__init__(tab)
            self.image = None
            self.imageQueues = ([], [])
            self.imageQueuesInd = 0
            self.curImageIndDisplayed = 0
            self.isDisplayingEpisode = False
            self.waitCount = 0
            self.renderImage = None
            self.trainingEpisodes = 0
            self.prevDisplayedEpisode = None

            self.curTotalEpisodes = None
            self.graphDataPoints = []
            self.curLossAccum = 0
            self.curRewardAccum = 0
            self.curEpisodeSteps = 0
            self.episodeAccLoss = 0
            self.episodeAccReward = 0
            self.episodeAccEpsilon = 0

            self.listener = listener

            tkinter.Label(self, text='Number of Episodes: ').grid(row=0, column=0)
            self.numEps = tkinter.Entry(self)
            self.numEps.grid(row=0, column=1)

            tkinter.Label(self, text='Learning Rate: ').grid(row=1, column=0)
            self.learningRate = tkinter.Scale(self, from_=0.01, to=1, resolution=0.01, orient=tkinter.HORIZONTAL)
            self.learningRate.grid(row=1, column=1)

            tkinter.Label(self, text='Max Steps: ').grid(row=2, column=0)
            self.maxSteps = tkinter.Entry(self)
            self.maxSteps.grid(row=2, column=1)

            tkinter.Label(self, text='Gamma: ').grid(row=3, column=0)
            self.gamma = tkinter.Scale(self, from_=0.00, to=1, resolution=0.01, orient=tkinter.HORIZONTAL)
            self.gamma.grid(row=3, column=1)

            tkinter.Label(self, text='Max Epsilon: ').grid(row=4, column=0)
            self.maxEpsilon = tkinter.Scale(self, from_=0.00, to=1, resolution=0.01, orient=tkinter.HORIZONTAL)
            self.maxEpsilon.grid(row=4, column=1)

            tkinter.Label(self, text='Min Epsilon: ').grid(row=5, column=0)
            self.minEpsilon = tkinter.Scale(self, from_=0.00, to=1, resolution=0.01, orient=tkinter.HORIZONTAL)
            self.minEpsilon.grid(row=5, column=1)

            tkinter.Label(self, text='Decay Rate: ').grid(row=6, column=0)
            self.decayRate = tkinter.Scale(self, from_=0.0, to=0.2, resolution=0.001, orient=tkinter.HORIZONTAL)
            self.decayRate.grid(row=6, column=1)

            self.slowLabel = tkinter.Label(self, text='Displayed episode speed')
            self.slowLabel.grid(row=7, column=0)
            self.slowSlider = tkinter.Scale(self, from_=1, to=20, resolution=1, orient=tkinter.HORIZONTAL)
            self.slowSlider.grid(row=7, column=1)

            self.trainButton = tkinter.Button(self, text='Train', fg='black', command=self.train)
            self.trainButton.grid(row=8, column=0)

            self.haltButton = tkinter.Button(self, text='Halt', fg='black', command=self.halt)
            self.haltButton.grid(row=8, column=1)

            self.resetButton = tkinter.Button(self, text='Reset Agent', fg='black', command=self.reset)
            self.resetButton.grid(row=9, column=0, columnspan=2)

            self.render = tkinter.Canvas(self)
            self.render.grid(row=0, column=2, rowspan=9, columnspan=8, sticky='wens')

            self.displayedEpisodeNum = tkinter.Label(self, text='')
            self.displayedEpisodeNum.grid(row=9, column=2)

            self.curEpisodeNum = tkinter.Label(self, text='')
            self.curEpisodeNum.grid(row=9, column=3)

            self.graph = tkinter.Canvas(self)
            self.graph.grid(row=10, column=2, rowspan=4, columnspan=8, sticky='wens')

        def halt(self):
            self.listener.halt()

        def reset(self):
            if self.listener.reset():
                self.trainingEpisodes = 0
                self.curEpisodeNum.configure(text='')
                self.displayedEpisodeNum.configure(text='')

                self.curTotalEpisodes = None
                self.graphDataPoints.clear()
                self.curLossAccum = 0
                self.curRewardAccum = 0
                self.curEpisodeSteps = 0
                self.episodeAccLoss = 0
                self.episodeAccReward = 0
                self.episodeAccEpsilon = 0
                self.graph.delete('all')

        def train(self):
            self.curTotalEpisodes = None
            self.graphDataPoints.clear()
            self.curLossAccum = 0
            self.curRewardAccum = 0
            self.curEpisodeSteps = 0
            self.episodeAccLoss = 0
            self.episodeAccReward = 0
            self.episodeAccEpsilon = 0
            self.graph.delete('all')

            try:
                total_episodes = int(self.numEps.get())
                learning_rate = self.learningRate.get()
                max_steps = int(self.maxSteps.get())
                gamma = self.gamma.get()
                max_epsilon = self.maxEpsilon.get()
                min_epsilon = self.minEpsilon.get()
                decay_rate = self.decayRate.get()

                self.curTotalEpisodes = total_episodes
                self.listener.startTraining(total_episodes,
                                            learning_rate,
                                            max_steps,
                                            gamma,
                                            max_epsilon,
                                            min_epsilon,
                                            decay_rate)

                self.checkMessages()
            except ValueError:
                print('Bad Hyperparameters')

        def checkMessages(self):
            while self.listener.messageQueue.qsize():
                message = self.listener.messageQueue.get(timeout=0)
                if message.type == Model.Message.EVENT:
                    if message.data == Model.Message.EPISODE:
                        self.addEpisodeToGraph()
                        self.trainingEpisodes += 1
                        self.curEpisodeNum.configure(text='Episodes completed: '+str(self.trainingEpisodes))
                        if self.isDisplayingEpisode:
                            self.imageQueues[self.imageQueuesInd].clear()
                        else:
                            self.imageQueuesInd = 1 - self.imageQueuesInd
                            self.imageQueues[self.imageQueuesInd].clear()
                            self.isDisplayingEpisode = True
                            self.displayedEpisodeNum.configure(text='Showing episode '+str(self.trainingEpisodes))
                    elif message.data == Model.Message.FINISHED:
                        self.imageQueues[0].clear()
                        self.imageQueues[1].clear()
                        self.imageQueuesInd = 0
                        self.curImageIndDisplayed = 0
                        self.isDisplayingEpisode = False
                        self.waitCount = 0
                        return
                elif message.type == Model.Message.STATE:
                    self.imageQueues[self.imageQueuesInd].append(message.data.image)
                    self.accumulateState(message.data)

            self.updateEpisodeRender()
            self.master.after(20, self.checkMessages)

        def addEpisodeToGraph(self):
            avgLoss = self.episodeAccLoss/self.curEpisodeSteps
            totalReward = self.episodeAccReward
            avgEpsilon = self.episodeAccEpsilon/self.curEpisodeSteps

            avgState = (avgLoss, totalReward, avgEpsilon)
            self.graphDataPoints.append(avgState)

            smoothAmt = 50

            w = self.graph.winfo_width()
            h = self.graph.winfo_height()

            oldX = w * (len(self.graphDataPoints) - 1) / self.curTotalEpisodes
            newX = w * (len(self.graphDataPoints)) / self.curTotalEpisodes

            if len(self.graphDataPoints) > 1:
                _, _, prevEpsilon = self.graphDataPoints[-2]
                oldY = h * (1 - prevEpsilon)
                newY = h * (1 - avgEpsilon)
                self.graph.create_line(oldX, oldY, newX, newY, fill='green')

            if len(self.graphDataPoints) > smoothAmt:

                prevLoss, prevReward = self.curLossAccum/smoothAmt, self.curRewardAccum/smoothAmt
                (obsLoss, obsReward, _) = self.graphDataPoints[-smoothAmt-1]

                self.curLossAccum -= obsLoss
                self.curRewardAccum -= obsReward
                self.curLossAccum += avgLoss
                self.curRewardAccum += totalReward

                curReward = self.curRewardAccum/smoothAmt
                curLoss = self.curLossAccum/smoothAmt

                oldY = h - prevReward*2
                newY = h - curReward*2
                self.graph.create_line(oldX, oldY, newX, newY, fill='red')

                oldY = h*(1 - prevLoss/40)
                newY = h*(1 - curLoss/40)
                self.graph.create_line(oldX, oldY, newX, newY, fill='blue')
            else:
                self.curLossAccum += avgLoss
                self.curRewardAccum += totalReward

            self.curEpisodeSteps = 0
            self.episodeAccLoss = 0
            self.episodeAccReward = 0
            self.episodeAccEpsilon = 0

        def accumulateState(self, state):
            self.episodeAccEpsilon += state.epsilon
            self.episodeAccReward += state.reward
            self.episodeAccLoss += state.loss
            self.curEpisodeSteps += 1

        def updateEpisodeRender(self):
            displayQueue = self.imageQueues[1 - self.imageQueuesInd]
            if displayQueue:
                if self.waitCount >= 21 - self.slowSlider.get():
                    self.waitCount = 0
                    tempImage = displayQueue[self.curImageIndDisplayed]
                    self.curImageIndDisplayed = self.curImageIndDisplayed+1
                    if self.curImageIndDisplayed == len(displayQueue):
                        self.curImageIndDisplayed = 0
                        self.isDisplayingEpisode = False
                    tempImage = tempImage.resize((self.render.winfo_width(), self.render.winfo_height()))
                    self.image = ImageTk.PhotoImage(tempImage) # must maintain a reference to this image in self: otherwise will be garbage collected
                    if self.renderImage:
                        self.render.delete(self.renderImage)
                    self.renderImage = self.render.create_image(0, 0, anchor='nw', image=self.image)
                self.waitCount += 1