import tkinter
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk
import ttkwidgets

from Agents import qLearning, drqn, deepQ, adrqn
from Agents.DeepSARSA import DeepSARSA
from Environments import cartPoleEnv, cartPoleEnvDiscrete, atariEnv, frozenLakeEnv, pendulumEnv, acrobotEnv, mountainCarEnv
from MVC.model import Model
from Agents.sarsa import sarsa
import math
import importlib.util

class View:
    agents = [deepQ.DeepQ, qLearning.QLearning, drqn.DRQN, adrqn.ADRQN, sarsa, DeepSARSA]
    environments = [cartPoleEnv.CartPoleEnv, cartPoleEnvDiscrete.CartPoleEnvDiscrete, frozenLakeEnv.FrozenLakeEnv,
                    pendulumEnv.PendulumEnv, acrobotEnv.AcrobotEnv, mountainCarEnv.MountainCarEnv]
    environments += atariEnv.AtariEnv.subEnvs

    """
    :param master: the top level widget of Tk
    :type master: tkinter.Tk
    :param listener: the listener object that will handle user input
    :type listener: controller.ViewListener
    """
    def __init__(self, master, listener):
        View.ProjectWindow(master, listener)


    class Window:
        def __init__(self, master, listener):
            self.master = master
            self.listener = listener
            self.frame = ttk.Frame(master)
            for i in range(10):
                self.frame.grid_columnconfigure(i, minsize=75)
                self.frame.grid_rowconfigure(i, minsize=50)

        def goBack(self):
            self.frame.destroy()


    class ProjectWindow(Window):
        def __init__(self, master, listener):
            super().__init__(master, listener)

            self.listener = listener
            self.tabIDCounter = 0
            self.closeTabButton = ttk.Button(self.frame, text='Close Current Tab', command=self.closeTab)
            self.closeTabButton.grid(row=0, column=0)
            self.rechooseButton = ttk.Button(self.frame, text='Reset Current Tab', command=self.rechoose)
            self.rechooseButton.grid(row=0, column=1)
            self.loadEnvButton = ttk.Button(self.frame, text='Load Environment', command=self.loadEnv)
            self.loadEnvButton.grid(row=0, column=2)
            self.tab = ttk.Notebook(self.frame)
            self.tab.bind("<<NotebookTabChanged>>", self.tabChange)

            self.tabs = [View.GeneralTab(self.tab, listener, self.tabIDCounter)]

            for tab in self.tabs:
                self.tab.add(tab, text='Tab '+str(self.tabIDCounter + 1))
                self.tabIDCounter += 1
            addTab = ttk.Frame(self.tab)
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

        def rechoose(self):
            tkId = self.tab.select()
            curTab = self.tab.nametowidget(tkId)
            if not curTab.listener.modelIsRunning(curTab.tabID):
                curTab.parameterFrame.destroy()
                curTab.parameterFrame = View.GeneralTab.ModelChooser(curTab)
                curTab.parameterFrame.grid(row=2, column=0, columnspan=2)

        def loadEnv(self):
            filename = filedialog.askopenfilename(initialdir="/", title="Select file")

            spec = importlib.util.spec_from_file_location("customenv", filename)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            View.environments = [mod.CustomEnv] + View.environments

            for ind, tab in enumerate(self.tabs):
                if isinstance(tab, View.GeneralTab) and isinstance(tab.parameterFrame, View.GeneralTab.ModelChooser):
                    tab.parameterFrame.destroy()
                    tab.parameterFrame = View.GeneralTab.ModelChooser(tab)
                    tab.parameterFrame.grid(row=2, column=0, columnspan=2)

    class GeneralTab(ttk.Frame):
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
            self.rewardGraphMin = 0
            self.rewardGraphMax = 100
            self.lossGraphMax = 100
            self.graphBottomMargin = 50

            self.listener = listener

            ttk.Label(self, text='Number of Episodes: ').grid(row=0, column=0)
            self.numEpsVar = tkinter.StringVar()
            self.numEps = ttk.Entry(self, textvariable=self.numEpsVar)
            self.numEpsVar.set('1000')
            self.numEps.grid(row=0, column=1)

            ttk.Label(self, text='Max Steps: ').grid(row=1, column=0)
            self.maxStepsVar = tkinter.StringVar()
            self.maxSteps = ttk.Entry(self, textvariable=self.maxStepsVar)
            self.maxStepsVar.set('200')
            self.maxSteps.grid(row=1, column=1)

            # Add model parameters here
            self.parameterFrame = self.ModelChooser(self)
            self.parameterFrame.grid(row=2, column=0, columnspan=2)

            self.slowLabel = ttk.Label(self, text='Displayed episode speed')
            self.slowLabel.grid(row=7, column=0)
            self.slowSlider = ttkwidgets.tickscale.TickScale(self, from_=1, to=20, resolution=1, orient=tkinter.HORIZONTAL)
            self.slowSlider.set(10)
            self.slowSlider.grid(row=7, column=1)
            
            self.render = tkinter.Canvas(self, background='#eff0f1')
            self.render.grid(row=0, column=2, rowspan=9, columnspan=8, sticky='wens')

            self.displayedEpisodeNum = ttk.Label(self, text='')
            self.displayedEpisodeNum.grid(row=9, column=2)

            self.curEpisodeNum = ttk.Label(self, text='')
            self.curEpisodeNum.grid(row=9, column=3)

            self.graph = tkinter.Canvas(self, background='#eff0f1')
            self.graph.grid(row=10, column=2, rowspan=4, columnspan=8, sticky='wens')
            self.graphLine = self.graph.create_line(0,0,0,0, fill='black')
            self.graph.bind("<Motion>", self.updateGraphLine)

            self.legend = tkinter.Canvas(self, background='#eff0f1')
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
            self.graph.coords(self.graphLine, [xVal, 0, xVal, height])

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

                    self.listener.startTesting(self.tabID, [total_episodes, max_steps] + self.parameterFrame.getParameters())
                    self.trainingEpisodes = 0
                    self.curTotalEpisodes = total_episodes
                    self.resetGraph()
                    self.checkMessages()
                    self.legend.itemconfig(self.testResult1, text='')
                    self.legend.itemconfig(self.testResult2, text='')
                except ValueError:
                    print('Bad Hyperparameters')

        def save(self):
            if not self.listener.modelIsRunning(self.tabID):
                filename = filedialog.asksaveasfilename(initialdir = "/",title = "Select file")
                self.listener.save(filename, self.tabID)

        def load(self):
            if not self.listener.modelIsRunning(self.tabID):
                filename = filedialog.askopenfilename(initialdir = "/",title = "Select file")
                self.listener.load(filename, self.tabID)

        def reset(self):
            if not self.listener.modelIsRunning(self.tabID):
                self.listener.reset(self.tabID)

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
            self.redrawGraphXAxis()

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

        def redrawGraphXAxis(self):
            w = self.graph.winfo_width()
            h = self.graph.winfo_height()

            step = 1
            while self.curTotalEpisodes // step > 13:
                step *= 5
                if self.curTotalEpisodes // step <= 13:
                    break
                step *= 2
            for ind in range(0, self.curTotalEpisodes, step):
                x = w * (ind / self.curTotalEpisodes)
                self.graph.create_line(x, h - self.graphBottomMargin, x, h - self.graphBottomMargin / 2)
                self.graph.create_text(x, h - self.graphBottomMargin / 2, text=str(ind), anchor='n')

        def redrawGraph(self, full):
            if full:
                lastN = len(self.graphDataPoints)
                self.curLossAccum = 0
                self.curRewardAccum = 0
                self.smoothedDataPoints.clear()
                self.lossGraphMax = max(0.0000000000001, sorted([loss for loss, _, _ in self.graphDataPoints])[int((len(self.graphDataPoints)-1)*0.95)]*1.1)
                rewardSorted = sorted([reward for _, reward, _ in self.graphDataPoints])
                self.rewardGraphMax = rewardSorted[int((len(self.graphDataPoints)-1)*0.95)]
                self.rewardGraphMin = rewardSorted[int((len(self.graphDataPoints)-1)*0.05)]
                extendAmt = 0.1*(self.rewardGraphMax - self.rewardGraphMin)
                self.rewardGraphMax += extendAmt
                self.rewardGraphMin -= extendAmt

                print('loss graph max:', self.lossGraphMax)
                print('reward graph min/max:', self.rewardGraphMin, self.rewardGraphMax)
                self.graph.delete('all')
                self.redrawGraphXAxis()
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

                    rewardRange = max(0.000000001, self.rewardGraphMax - self.rewardGraphMin)
                    oldY = self.graphBottomMargin + (h - self.graphBottomMargin) * (1 - (prevReward - self.rewardGraphMin)/rewardRange)
                    newY = self.graphBottomMargin + (h - self.graphBottomMargin) * (1 - (curReward - self.rewardGraphMin)/rewardRange)
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
            for agent in View.agents:
                if self.parameterFrame.agentOpts.get() == agent.displayName:
                    break
            for env in View.environments:
                if self.parameterFrame.envOpts.get() == env.displayName:
                    break
            self.parameterFrame.destroy()
            self.parameterFrame = self.ParameterFrame(self, agent, env)
            self.parameterFrame.grid(row=2, column=0, columnspan=2)

        def close(self):
            self.listener.close(self.tabID)

        class ParameterFrame(ttk.Frame):
            def __init__(self, master, agentClass, envClass):
                super().__init__(master)
                self.master = master
                master.listener.setAgent(master.tabID, agentClass)
                master.listener.setEnvironment(master.tabID, envClass)
                self.values = []
                for param in agentClass.parameters:
                    subFrame = ttk.Frame(self)
                    ttk.Label(subFrame, text=param.name).pack(side='left')
                    scale = ttkwidgets.tickscale.TickScale(subFrame, from_=param.min, to=param.max, resolution=param.resolution,
                                          orient=tkinter.HORIZONTAL)
                    scale.set(param.default)
                    scale.pack(side='left')
                    subFrame.pack()
                    self.values.append(scale)
                ttk.Button(self, text='Train', command=self.master.train).pack(side='left')
                ttk.Button(self, text='Halt', command=self.master.halt).pack(side='left')
                ttk.Button(self, text='Test', command=self.master.test).pack(side='left')
                ttk.Button(self, text='Save', command=self.master.save).pack(side='left')
                ttk.Button(self, text='Load', command=self.master.load).pack(side='left')
                ttk.Button(self, text='Reset', command=self.master.reset).pack(side='left')

            def getParameters(self):
                return [value.get() for value in self.values]

        class ModelChooser(ttk.Frame):
            def __init__(self, master):
                super().__init__(master)
                self.agentOpts = tkinter.StringVar(self)
                self.envOpts = tkinter.StringVar(self)
                subFrame = ttk.Frame(self)

                value = [opt.displayName for opt in View.environments]
                value2 = [opt.displayName for opt in View.agents]

                ttk.Combobox(subFrame, state='readonly', values=value2, textvariable = self.agentOpts).pack(side='left')
                ttk.Combobox(subFrame, state='readonly', values=value, textvariable = self.envOpts).pack(side='left')
                self.agentOpts.set('Select Agent')
                self.envOpts.set('Select Environment')

                subFrame.pack()
                ttk.Button(self, text='Set Model', command=master.selectModel).pack()

        class EnvironmentChooser(ttk.Frame):

            def __init__(self, master, listener):
                super().__init__(master, listener)

                self.title = ttk.Label(self.frame, text='Select an Environment:')
                self.title.grid(row=1, column=4, columnspan=2, sticky='wens')

                self.frozenLakeButton = ttk.Button(self.frame, text='Frozen Lake', fg='black',
                                                       command=self.chooseFrozenLake)
                self.frozenLakeButton.grid(row=2, column=4, columnspan=2, sticky='wens')

                self.cartPoleButton = ttk.Button(self.frame, text='Cart Pole', fg='black',
                                                     command=self.chooseCartPoleEnv)
                self.cartPoleButton.grid(row=3, column=4, columnspan=2, sticky='wens')

                self.cartPoleDiscreteButton = ttk.Button(self.frame, text='Cart Pole Discretized', fg='black',
                                                             command=self.chooseCartPoleDiscreteEnv)
                self.cartPoleDiscreteButton.grid(row=4, column=4, columnspan=2, sticky='wens')

                self.customButton = ttk.Button(self.frame, text='Custom Environment', fg='black',
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
