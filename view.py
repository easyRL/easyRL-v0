import tkinter
from tkinter import ttk
from PIL import Image, ImageTk

import agent
import deepQ, qLearning, drqn, adrqn
import valueIteration
import cartPoleEnv, cartPoleEnvDiscrete, frozenLakeEnv, alien
from model import Model
from sarsa import sarsa

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

        def goBack(self):
            self.frame.destroy()

    class StartWindow(Window):
        def __init__(self, master, listener):
            super().__init__(master, listener)

            self.projectsButton = tkinter.Button(self.frame, text='Load Agent', fg='black')
            self.projectsButton.grid(row=2, column=2, rowspan=2, columnspan = 2, sticky='wens')

            self.examplesButton = tkinter.Button(self.frame, text='Example Agents', fg='black')
            self.examplesButton.grid(row=2, column=6, rowspan=2, columnspan = 2, sticky='wens')

            self.newButton = tkinter.Button(self.frame, text='New Agent', fg='black', command=self.handleButton)
            self.newButton.grid(row=6, column=4, rowspan=2, columnspan = 2, sticky='wens')

            self.frame.grid(row=0, column=0)
            self.frame.lift()

        def handleButton(self):
            View.ProjectWindow(self.master, self.listener)

    class ProjectWindow(Window):
        def __init__(self, master, listener):
            super().__init__(master, listener)

            self.listener = listener
            self.tabIDCounter = 0
            self.backButton = tkinter.Button(self.frame, text='<- Back', fg='black', command=self.goBack)
            self.backButton.grid(row=0, column=0)
            self.closeTabButton = tkinter.Button(self.frame, text='Close Current Tab', fg='black', command=self.closeTab)
            self.closeTabButton.grid(row=0, column=1)

            self.tab = ttk.Notebook(self.frame)
            self.tab.bind("<<NotebookTabChanged>>", self.tabChange)

            self.tabs = [View.GeneralTab(self.tab, listener, self.tabIDCounter)]

            for tab in self.tabs:
                self.tab.add(tab, text='Tab '+str(self.tabIDCounter + 1))
                self.tabIDCounter += 1
            addTab = tkinter.Frame(self.tab)
            self.tab.add(addTab, text='+')
            self.tabs.append(addTab)

            self.tab.grid(row=1, column=0, rowspan=9, columnspan=10, sticky='wens')

            self.frame.grid(row=0, column=0)
            self.frame.lift()

        def tabChange(self, event):
            tabIndex = event.widget.index('current')
            if len(self.tabs) > 1 and tabIndex == len(self.tabs)-1:
                newTab = View.GeneralTab(self.tab, self.listener, self.tabIDCounter)
                self.tab.forget(self.tabs[-1])
                self.tab.add(newTab, text='Tab '+str(self.tabIDCounter+1))
                self.tab.add(self.tabs[-1], text='+')
                self.tabs = self.tabs[:-1] + [newTab] + [self.tabs[-1]]
                self.tab.select(newTab)
                self.tabIDCounter += 1

        def closeTab(self):
            if len(self.tabs) != 2:
                tkId = self.tab.select()
                curTab = self.tab.nametowidget(tkId)
                curTab.close()
                ind = 0
                while self.tabs[ind] != curTab:
                    ind += 1
                self.tabs = self.tabs[:ind] + self.tabs[ind + 1:]
                if ind == len(self.tabs)-1:
                    self.tab.select(self.tabs[-2])
                self.tab.forget(tkId)
                self.tabIDCounter = self.tabs[-2].tabID+1

        def goBack(self):
            for tab in self.tabs[:-1]:
                tab.close()
            super().goBack()

    class GeneralTab(tkinter.Frame):
        def __init__(self, tab, listener, tabID):
            super().__init__(tab)
            self.tabID = tabID
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
            self.smoothedDataPoints = []
            self.curLossAccum = 0
            self.curRewardAccum = 0
            self.curEpisodeSteps = 0
            self.episodeAccLoss = 0
            self.episodeAccReward = 0
            self.episodeAccEpsilon = 0

            self.smoothAmt = 20
            self.rewardGraphMax = 100
            self.lossGraphMax = 100
            self.graphLineCoords = None

            self.listener = listener

            tkinter.Label(self, text='Number of Episodes: ').grid(row=0, column=0)
            numEpsVar = tkinter.StringVar()
            self.numEps = tkinter.Entry(self, textvariable=numEpsVar)
            numEpsVar.set('1000')
            self.numEps.grid(row=0, column=1)

            tkinter.Label(self, text='Max Steps: ').grid(row=1, column=0)
            maxStepsVar = tkinter.StringVar()
            self.maxSteps = tkinter.Entry(self, textvariable=maxStepsVar)
            maxStepsVar.set('200')
            self.maxSteps.grid(row=1, column=1)

            # Add model parameters here
            self.parameterFrame = self.ModelChooser(self)
            self.parameterFrame.grid(row=2, column=0, columnspan=2)

            self.slowLabel = tkinter.Label(self, text='Displayed episode speed')
            self.slowLabel.grid(row=7, column=0)
            self.slowSlider = tkinter.Scale(self, from_=1, to=20, resolution=1, orient=tkinter.HORIZONTAL)
            self.slowSlider.set(10)
            self.slowSlider.grid(row=7, column=1)

            self.render = tkinter.Canvas(self)
            self.render.grid(row=0, column=2, rowspan=9, columnspan=8, sticky='wens')

            self.displayedEpisodeNum = tkinter.Label(self, text='')
            self.displayedEpisodeNum.grid(row=9, column=2)

            self.curEpisodeNum = tkinter.Label(self, text='')
            self.curEpisodeNum.grid(row=9, column=3)

            self.graph = tkinter.Canvas(self)
            self.graph.grid(row=10, column=2, rowspan=4, columnspan=8, sticky='wens')
            self.graphLine = self.graph.create_line(0,0,0,0, fill='black')
            self.graph.bind("<Motion>", self.updateGraphLine)

            self.legend = tkinter.Canvas(self)
            self.legend.grid(row=10, column=0, rowspan=4, columnspan=2, sticky='wens')
            self.legend.bind('<Configure>', self.legendResize)


        def legendResize(self, evt):
            self.legend.delete('all')
            h = evt.height
            p1, p2, p3, p4, p5 = h/5, 2*h/5, 3*h/5, 4*h/5, 9*h/10
            self.legend.create_line(40, p1, 90, p1, fill='blue')
            self.legend.create_line(40, p2, 90, p2, fill='red')
            self.legend.create_line(40, p3, 90, p3, fill='green')
            self.lossLegend = self.legend.create_text(100, p1, text='MSE Episode Loss:', anchor='w')
            self.rewardLegend = self.legend.create_text(100, p2, text='Episode Reward:', anchor='w')
            self.epsilonLegend = self.legend.create_text(100, p3, text='Epsilon:', anchor='w')
            self.testResult1 = self.legend.create_text(100,p4, text='', anchor='w')
            self.testResult2 = self.legend.create_text(100,p5, text='', anchor='w')

        def updateGraphLine(self, evt):
            xVal = evt.x
            height = self.graph.winfo_height()
            self.graphLineCoords = [xVal, 0, xVal, height]

            if self.curTotalEpisodes:
                smoothIndex = (int)(self.curTotalEpisodes*xVal/self.graph.winfo_width())-self.smoothAmt
                if len(self.smoothedDataPoints) > smoothIndex >= 0:
                    loss, reward, epsilon = self.smoothedDataPoints[smoothIndex]
                    self.legend.itemconfig(self.lossLegend, text='MSE Episode Loss: '+str(loss))
                    self.legend.itemconfig(self.rewardLegend, text='Episode Reward: '+str(reward))
                    self.legend.itemconfig(self.epsilonLegend, text='Epsilon: '+str(epsilon))
                else:
                    self.legend.itemconfig(self.lossLegend, text='MSE Episode Loss:')
                    self.legend.itemconfig(self.rewardLegend, text='Episode Reward:')
                    self.legend.itemconfig(self.epsilonLegend, text='Epsilon:')

        def halt(self):
            self.listener.halt(self.tabID)
            self.imageQueues[0].clear()
            self.imageQueues[1].clear()
            self.imageQueuesInd = 0
            self.curImageIndDisplayed = 0
            self.isDisplayingEpisode = False
            self.waitCount = 0

        def train(self):
            if not self.listener.modelIsRunning(self.tabID):
                try:
                    total_episodes = int(self.numEps.get())
                    max_steps = int(self.maxSteps.get())

                    self.listener.startTraining(self.tabID, [total_episodes, max_steps] + self.parameterFrame.getParameters())
                    self.trainingEpisodes = 0
                    self.curTotalEpisodes = total_episodes
                    self.resetGraph()
                    self.checkMessages()
                    self.legend.itemconfig(self.testResult1, text='')
                    self.legend.itemconfig(self.testResult2, text='')
                except ValueError:
                    print('Bad Hyperparameters')

        def test(self):
            if not self.listener.modelIsRunning(self.tabID):
                try:
                    total_episodes = int(self.numEps.get())
                    max_steps = int(self.maxSteps.get())

                    self.listener.startTesting(self.tabID, [total_episodes, max_steps])
                    self.trainingEpisodes = 0
                    self.curTotalEpisodes = total_episodes
                    self.resetGraph()
                    self.checkMessages()
                    self.legend.itemconfig(self.testResult1, text='')
                    self.legend.itemconfig(self.testResult2, text='')
                except ValueError:
                    print('Bad Hyperparameters')

        def resetGraph(self):
            self.graphDataPoints.clear()
            self.smoothedDataPoints.clear()
            self.curLossAccum = 0
            self.curRewardAccum = 0
            self.curEpisodeSteps = 0
            self.episodeAccLoss = 0
            self.episodeAccReward = 0
            self.episodeAccEpsilon = 0
            self.graph.delete('all')
            self.graphLine = self.graph.create_line(0, 0, 0, 0, fill='black')

        def checkMessages(self):
            while self.listener.getQueue(self.tabID).qsize():
                message = self.listener.getQueue(self.tabID).get(timeout=0)
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
                            self.curImageIndDisplayed = 0
                            self.displayedEpisodeNum.configure(text='Showing episode '+str(self.trainingEpisodes))
                    elif message.data == Model.Message.TRAIN_FINISHED:
                        self.imageQueues[0].clear()
                        self.imageQueues[1].clear()
                        self.imageQueuesInd = 0
                        self.curImageIndDisplayed = 0
                        self.isDisplayingEpisode = False
                        self.waitCount = 0
                        totalReward = sum([reward for _, reward, _ in self.graphDataPoints])
                        avgReward = totalReward/len(self.graphDataPoints)
                        self.legend.itemconfig(self.testResult1, text='Total Training Reward: '+str(totalReward))
                        self.legend.itemconfig(self.testResult2, text='Reward/Episode: '+str(avgReward))
                        return
                    elif message.data == Model.Message.TEST_FINISHED:
                        self.imageQueues[0].clear()
                        self.imageQueues[1].clear()
                        self.imageQueuesInd = 0
                        self.curImageIndDisplayed = 0
                        self.isDisplayingEpisode = False
                        self.waitCount = 0
                        totalReward = sum([reward for _, reward, _ in self.graphDataPoints])
                        avgReward = totalReward/len(self.graphDataPoints)
                        self.legend.itemconfig(self.testResult1, text='Total Test Reward: '+str(totalReward))
                        self.legend.itemconfig(self.testResult2, text='Reward/Episode: '+str(avgReward))
                        return
                elif message.type == Model.Message.STATE:
                    self.imageQueues[self.imageQueuesInd].append(message.data.image)
                    self.accumulateState(message.data)

            self.updateEpisodeRender()
            if self.graphLineCoords:
                self.graph.coords(self.graphLine, self.graphLineCoords)
            self.master.after(10, self.checkMessages)

        def addEpisodeToGraph(self):
            avgLoss = self.episodeAccLoss/self.curEpisodeSteps
            totalReward = self.episodeAccReward
            avgEpsilon = self.episodeAccEpsilon/self.curEpisodeSteps

            avgState = (avgLoss, totalReward, avgEpsilon)
            self.graphDataPoints.append(avgState)

            self.redrawGraph(len(self.graphDataPoints)%20 == 0)

            self.curEpisodeSteps = 0
            self.episodeAccLoss = 0
            self.episodeAccReward = 0
            self.episodeAccEpsilon = 0

        def redrawGraph(self, full):
            if full:
                lastN = len(self.graphDataPoints)
                self.curLossAccum = 0
                self.curRewardAccum = 0
                self.smoothedDataPoints.clear()
                self.lossGraphMax = max(0.0000000000001, sorted([loss for loss, _, _ in self.graphDataPoints])[int((len(self.graphDataPoints)-1)*0.95)]*1.1)
                self.rewardGraphMax = max(0.0000000000001, sorted([reward for _, reward, _ in self.graphDataPoints])[int((len(self.graphDataPoints)-1)*0.95)]*1.1)

                print('loss graph max:', self.lossGraphMax)
                print('reward graph max:', self.rewardGraphMax)
                self.graph.delete('all')
                self.graphLine = self.graph.create_line(0, 0, 0, 0, fill='black')
            else:
                lastN = 1

            w = self.graph.winfo_width()
            h = self.graph.winfo_height()

            offset = len(self.graphDataPoints) - lastN
            for ind in range(max(0,offset), len(self.graphDataPoints)):
                oldX = w * (ind / self.curTotalEpisodes)
                newX = w * ((ind+1) / self.curTotalEpisodes)
                avgLoss, totalReward, avgEpsilon = self.graphDataPoints[ind]
                if ind > 0:
                    _, _, prevEpsilon = self.graphDataPoints[ind-1]
                    oldY = h * (1 - prevEpsilon)
                    newY = h * (1 - avgEpsilon)
                    self.graph.create_line(oldX, oldY, newX, newY, fill='green')

                if ind >= self.smoothAmt:
                    prevLoss, prevReward = self.curLossAccum/self.smoothAmt, self.curRewardAccum/self.smoothAmt
                    (obsLoss, obsReward, _) = self.graphDataPoints[ind-self.smoothAmt]

                    self.curLossAccum -= obsLoss
                    self.curRewardAccum -= obsReward
                    self.curLossAccum += avgLoss
                    self.curRewardAccum += totalReward

                    curReward = self.curRewardAccum/self.smoothAmt
                    curLoss = self.curLossAccum/self.smoothAmt
                    self.smoothedDataPoints.append((curLoss, curReward, avgEpsilon))

                    oldY = h*(1 - prevReward/self.rewardGraphMax)
                    newY = h*(1 - curReward/self.rewardGraphMax)
                    self.graph.create_line(oldX, oldY, newX, newY, fill='red')

                    oldY = h*(1 - prevLoss/self.lossGraphMax)
                    newY = h*(1 - curLoss/self.lossGraphMax)
                    self.graph.create_line(oldX, oldY, newX, newY, fill='blue')
                else:
                    self.curLossAccum += avgLoss
                    self.curRewardAccum += totalReward

        def accumulateState(self, state):
            if state.epsilon:
                self.episodeAccEpsilon += state.epsilon
            if state.reward:
                self.episodeAccReward += state.reward
            if state.loss:
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

        def selectModel(self):
            for agent in self.parameterFrame.agents:
                if self.parameterFrame.agentOpts.get() == agent.displayName:
                    break
            for env in self.parameterFrame.environments:
                if self.parameterFrame.envOpts.get() == env.displayName:
                    break
            self.parameterFrame.destroy()
            self.parameterFrame = self.ParameterFrame(self, agent, env)
            self.parameterFrame.grid(row=2, column=0, columnspan=2)

        def close(self):
            self.listener.close(self.tabID)

        class ParameterFrame(tkinter.Frame):
            def __init__(self, master, agentClass, envClass):
                super().__init__(master)
                self.master = master
                master.listener.setAgent(master.tabID, agentClass)
                master.listener.setEnvironment(master.tabID, envClass)
                self.values = []
                for param in agentClass.parameters:
                    subFrame = tkinter.Frame(self)
                    tkinter.Label(subFrame, text=param.name).pack(side='left')
                    scale = tkinter.Scale(subFrame, from_=param.min, to=param.max, resolution=param.resolution,
                                          orient=tkinter.HORIZONTAL)
                    scale.set(param.default)
                    scale.pack(side='left')
                    subFrame.pack()
                    self.values.append(scale)
                tkinter.Button(self, text='Train', fg='black', command=self.master.train).pack(side='left')
                tkinter.Button(self, text='Halt', fg='black', command=self.master.halt).pack(side='left')
                tkinter.Button(self, text='Test', fg='black', command=self.master.test).pack(side='left')

            def getParameters(self):
                return [value.get() for value in self.values]

        class ModelChooser(tkinter.Frame):
            agents = [deepQ.DeepQ, qLearning.QLearning, drqn.DRQN, adrqn.ADRQN]
            environments = [cartPoleEnv.CartPoleEnv, cartPoleEnvDiscrete.CartPoleEnvDiscrete, frozenLakeEnv.FrozenLakeEnv, alien.AlienEnv]

            def __init__(self, master):
                super().__init__(master)
                self.agentOpts = tkinter.StringVar(self)
                self.envOpts = tkinter.StringVar(self)
                subFrame = tkinter.Frame(self)
                tkinter.OptionMenu(subFrame, self.agentOpts, *[opt.displayName for opt in self.agents]).pack(side='left')
                tkinter.OptionMenu(subFrame, self.envOpts, *[opt.displayName for opt in self.environments]).pack(side='left')
                self.agentOpts.set(self.agents[0].displayName)
                self.envOpts.set(self.environments[0].displayName)
                subFrame.pack()
                tkinter.Button(self, text='Set Model', command=master.selectModel).pack()

        class EnvironmentChooser(tkinter.Frame):

            def __init__(self, master, listener):
                super().__init__(master, listener)

                self.title = tkinter.Label(self.frame, text='Select an Environment:')
                self.title.grid(row=1, column=4, columnspan=2, sticky='wens')

                self.frozenLakeButton = tkinter.Button(self.frame, text='Frozen Lake', fg='black',
                                                       command=self.chooseFrozenLake)
                self.frozenLakeButton.grid(row=2, column=4, columnspan=2, sticky='wens')

                self.cartPoleButton = tkinter.Button(self.frame, text='Cart Pole', fg='black',
                                                     command=self.chooseCartPoleEnv)
                self.cartPoleButton.grid(row=3, column=4, columnspan=2, sticky='wens')

                self.cartPoleDiscreteButton = tkinter.Button(self.frame, text='Cart Pole Discretized', fg='black',
                                                             command=self.chooseCartPoleDiscreteEnv)
                self.cartPoleDiscreteButton.grid(row=4, column=4, columnspan=2, sticky='wens')

                self.customButton = tkinter.Button(self.frame, text='Custom Environment', fg='black',
                                                   command=self.chooseCustom)
                self.customButton.grid(row=5, column=4, columnspan=2, sticky='wens')

                self.frame.grid(row=0, column=0)
                self.frame.lift()

            def chooseFrozenLake(self):
                self.listener.setEnvironment()
                View.ProjectWindow(self.master, self.listener)
                self.frame.destroy()

            def chooseCartPoleEnv(self):
                self.listener.setCartPoleEnv()
                View.ProjectWindow(self.master, self.listener)
                self.frame.destroy()

            def chooseCartPoleDiscreteEnv(self):
                self.listener.setCartPoleDiscreteEnv()
                View.ProjectWindow(self.master, self.listener)
                self.frame.destroy()

            def chooseCustom(self):
                pass
