import tkinter
from tkinter import ttk
from tkinter import filedialog, W
from tkinter import messagebox
from tkinter.ttk import Style

from ttkthemes import ThemedTk
from PIL import Image
from PIL import ImageTk
from PIL.ImageTk import PhotoImage
import ttkwidgets

from Agents import qLearning, drqn, deepQ, adrqn, agent, doubleDuelingQNative, drqnNative, drqnConvNative, ppoNative, reinforceNative, actorCriticNative, cem, npg, ddpg, sac, ppo, trpo, rainbow
from Agents.Collections import qTable
from Environments import cartPoleEnv, cartPoleEnvDiscrete, atariEnv, frozenLakeEnv, pendulumEnv, acrobotEnv, \
    mountainCarEnv
from MVC import helptext
from MVC.model import Model
from Agents.sarsa import sarsa
import importlib.util

about = """
    software requirements:
    Our code can be run on Mac Linux or Windows PC Operating systems with Visual Studio C++ build tools
    Requires python 3.7 and # pytorch 1.6
    # Further requires Tensorflow 2.1, Keras, Kivy and other packages, see Readme.txt for an explanation 
        and requirements.txt for details.

    EasyRL was created by the following students at the university of washington tacoma: 

    Neil Hulbert, Sam Spillers, Brandon Francis, James Haines-Temons, Ken Gil Romero
    Sam Wong, Kevin Flora, Bowei Huang
        """


class View:
    agents = [deepQ.DeepQ, deepQ.DeepQPrioritized, deepQ.DeepQHindsight, qLearning.QLearning, drqn.DRQN, drqn.DRQNPrioritized, drqn.DRQNHindsight, adrqn.ADRQN, adrqn.ADRQNPrioritized, adrqn.ADRQNHindsight, doubleDuelingQNative.DoubleDuelingQNative, drqnNative.DRQNNative, drqnConvNative.DRQNConvNative, ppoNative.PPONative, reinforceNative.ReinforceNative, actorCriticNative.ActorCriticNative, sarsa, cem.CEM, npg.NPG, ddpg.DDPG, sac.SAC, trpo.TRPO, rainbow.Rainbow]
    singleDimEnvs = [cartPoleEnv.CartPoleEnv, cartPoleEnvDiscrete.CartPoleEnvDiscrete, frozenLakeEnv.FrozenLakeEnv,
                    pendulumEnv.PendulumEnv, acrobotEnv.AcrobotEnv, mountainCarEnv.MountainCarEnv]
    environments = singleDimEnvs + atariEnv.AtariEnv.subEnvs

    allowedEnvs = {
        deepQ.DeepQ: singleDimEnvs,
        deepQ.DeepQPrioritized: singleDimEnvs,
        deepQ.DeepQHindsight: singleDimEnvs,
        qLearning.QLearning: [cartPoleEnvDiscrete.CartPoleEnvDiscrete, frozenLakeEnv.FrozenLakeEnv],
        drqn.DRQN: environments,
        drqn.DRQNPrioritized: environments,
        drqn.DRQNHindsight: environments,
        adrqn.ADRQN: environments,
        adrqn.ADRQNPrioritized: environments,
        adrqn.ADRQNHindsight: environments,
        doubleDuelingQNative.DoubleDuelingQNative: singleDimEnvs,
        drqnNative.DRQNNative: singleDimEnvs,
        drqnConvNative.DRQNConvNative: atariEnv.AtariEnv.subEnvs,
        ppoNative.PPONative: singleDimEnvs,
        reinforceNative.ReinforceNative: singleDimEnvs,
        actorCriticNative.ActorCriticNative: singleDimEnvs,
        sarsa: [cartPoleEnvDiscrete.CartPoleEnvDiscrete, frozenLakeEnv.FrozenLakeEnv],
        trpo.TRPO: singleDimEnvs,
        rainbow.Rainbow: singleDimEnvs,
        cem.CEM: environments,
        npg.NPG: environments,
        ddpg.DDPG: environments,
        sac.SAC: environments
    }

    allowedEnvs = {agent.displayName:[env.displayName for env in envs] for (agent, envs) in allowedEnvs.items()}
    allowedAgents = {}
    for agent, envs in allowedEnvs.items():
        for env in envs:
            curAgents = allowedAgents.get(env)
            if not curAgents:
                curAgents = []
                allowedAgents[env] = curAgents
            curAgents.append(agent)

    """
    :param master: the top level widget of Tk
    :type master: tkinter.Tk
    :param listener: the listener object that will handle user input
    :type listener: controller.ViewListener
    """

    def __init__(self, listener):
        self.root = ThemedTk(theme='keramik')
        self.root.resizable(False, False)
        self.root.geometry('1100x1080')
        self.root.configure(bg="gray80")
        self.root.title('EasyRL')
        # self.root.attributes('-fullscreen', True)
        self.listener = listener
        pw = View.ProjectWindow(self.root, listener)

        self.menubar = tkinter.Menu(self.root)
        self.mMenuFile = tkinter.Menu(self.menubar, tearoff=0)
        self.mMenuFile.add_command(label="Load Agent", command=pw.loadAgent)
        self.mMenuFile.add_command(label="Load Environment", command=pw.loadEnv)
        self.mMenuFile.add_command(label="Close Tab", command=pw.closeTab, state=tkinter.DISABLED)
        pw.mMenuFile = self.mMenuFile
        self.mMenuFile.add_command(label="Reset Tab", command=pw.rechoose)
        self.mMenuFile.add_command(label="Save Model", command=pw.save)
        self.mMenuFile.add_command(label="Load Model", command=pw.load)
        self.mMenuFile.add_command(label="Save Results", command=pw.saveResults)
        self.mMenuFile.add_separator()
        self.mMenuFile.add_command(label="Exit", command=self.delete_window)
        self.menubar.add_cascade(label="File", menu=self.mMenuFile)
        self.mMenuRun = tkinter.Menu(self.menubar, tearoff=0)
        self.mMenuRun.add_command(label="Train", command=pw.train)
        self.mMenuRun.add_command(label="Halt", command=pw.halt)
        self.mMenuRun.add_command(label="Test", command=pw.test)
        self.mMenuRun.add_command(label="Reset", command=pw.reset)
        self.menubar.add_cascade(label="Run", menu=self.mMenuRun)
        self.mMenuHelp = tkinter.Menu(self.menubar, tearoff=0)
        self.mMenuHelp.add_command(label="Help", command=self.helpMenu)
        self.mMenuHelp.add_command(label="About", command=self.about)
        self.menubar.add_cascade(label="Help", menu=self.mMenuHelp)
        self.root.config(menu=self.menubar)

        center(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self.delete_window)
        self.root.mainloop()

    def about(self):
        popup = tkinter.Tk()
        popup.wm_title("About")
        popup.geometry("1000x1000")

        texts = about
        sbar = tkinter.Scrollbar(popup)
        sbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)

        text = tkinter.Text(popup, height=1000, width=1000)
        text.configure(yscrollcommand=sbar.set)
        text.pack(expand=0, fill=tkinter.BOTH)
        text.insert(tkinter.END, texts)
        sbar.config(command=text.yview)
        text.config(state="disabled")
        center(popup)
        popup.mainloop()

    def helpMenu(self):
        popup = tkinter.Tk()
        popup.wm_title("Help")
        popup.geometry("1000x1000")

        texts = helptext.getHelpGettingStarted()
        sbar = tkinter.Scrollbar(popup)
        sbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)

        text = tkinter.Text(popup, height=1000, width=1000)
        text.configure(yscrollcommand=sbar.set)
        text.pack(expand=0, fill=tkinter.BOTH)
        text.insert(tkinter.END, texts)
        sbar.config(command=text.yview)
        text.config(state="disabled")
        center(popup)
        popup.mainloop()

    def delete_window(self):
        self.listener.haltAll()
        try:
            self.root.destroy()
        except:
            pass

    class CreateToolTip(
        object):  # Source: https://stackoverflow.com/questions/3221956/how-do-i-display-tooltips-in-tkinter
        def __init__(self, widget, text='widget info'):
            self.waittime = 500  # miliseconds
            self.wraplength = 180  # pixels
            self.widget = widget
            self.text = text
            self.widget.bind("<Enter>", self.enter)
            self.widget.bind("<Leave>", self.leave)
            self.widget.bind("<ButtonPress>", self.leave)
            self.id = None
            self.tw = None

        def enter(self, event=None):
            self.schedule()

        def leave(self, event=None):
            self.unschedule()
            self.hidetip()

        def schedule(self):
            self.unschedule()
            self.id = self.widget.after(self.waittime, self.showtip)

        def unschedule(self):
            id = self.id
            self.id = None
            if id:
                self.widget.after_cancel(id)

        def showtip(self, event=None):
            x = y = 0
            x, y, cx, cy = self.widget.bbox("insert")
            x += self.widget.winfo_rootx() + 25
            y += self.widget.winfo_rooty() + 20
            # creates a toplevel window
            self.tw = tkinter.Toplevel(self.widget)
            # Leaves only the label and removes the app window
            self.tw.wm_overrideredirect(True)
            self.tw.wm_geometry("+%d+%d" % (x, y))
            label = tkinter.Label(self.tw, text=self.text, justify='left',
                                  background="#ffffff", relief='solid', borderwidth=1,
                                  wraplength=self.wraplength)
            label.pack(ipadx=1)

        def hidetip(self):
            tw = self.tw
            self.tw = None
            if tw:
                tw.destroy()

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
            self.master = master
            self.listener = listener
            self.tabIDCounter = 0
            # self.closeTabButton = ttk.Button(self.frame, text='Close Current Tab', command=self.closeTab)
            # self.closeTabButton.grid(row=0, column=0)
            # close_button_ttp = View.CreateToolTip(self.closeTabButton, "Close Current Tab")
            # self.rechooseButton = ttk.Button(self.frame, text='Reset Current Tab', command=self.rechoose)
            # self.rechooseButton.grid(row=0, column=1)
            # reset_button_ttp = View.CreateToolTip(self.rechooseButton, "Reset Current Tab")
            # self.loadEnvButton = ttk.Button(self.frame, text='Load Environment', command=self.loadEnv)
            # self.loadEnvButton.grid(row=0, column=2)
            # load_env_button_ttp = View.CreateToolTip(self.loadEnvButton, "Load Custom  Environment")
            # self.loadAgentButton = ttk.Button(self.frame, text='Load Agent', command=self.loadAgent)
            # self.loadAgentButton.grid(row=0, column=3)
            # load_agent_button_ttp = View.CreateToolTip(self.loadAgentButton, "Load Custom Agent")
            tempFrame = tkinter.Frame(self.frame)
            train = tkinter.Button(tempFrame, text='Train', command=self.train)
            train.pack(side='left')
            train_button_ttp = View.CreateToolTip(train, "Train the agent with the current settings")
            halt = tkinter.Button(tempFrame, text='Halt', command=self.halt)
            halt.pack(side='left')
            halt_button_ttp = View.CreateToolTip(halt, "Pause the current training")
            test = tkinter.Button(tempFrame, text='Test', command=self.test)
            test.pack(side='left')
            test_button_ttp = View.CreateToolTip(test, "Test the agent in its current state")
            save = tkinter.Button(tempFrame, text='Save Model', command=self.save)
            save.pack(side='left')
            load = tkinter.Button(tempFrame, text='Load Model', command=self.load)
            load.pack(side='left')
            save_button_ttp = View.CreateToolTip(save, "Save the model in its current state")
            # load = ttk.Button(tempFrame, text='Load Agent', command=self.loadAgent)
            # load.pack(side='left')
            # btnLoadEnv = ttk.Button(tempFrame, text='Load Environment', command=self.loadEnv)
            # btnLoadEnv.pack(side='left')
            load_button_ttp = View.CreateToolTip(load, "Load a model")
            reset = tkinter.Button(tempFrame, text='Reset', command=self.reset)
            reset.pack(side='left')
            reset_button_ttp = View.CreateToolTip(reset, "Reset the current agent and its parameters")
            save_results = tkinter.Button(tempFrame, text='Save Results', command=self.saveResults)
            save_results.pack(side='left')
            save_results_button_ttp = View.CreateToolTip(save_results,
                                                         "Save the results of the current training session")
            tempFrame.grid(row=0, column=0, columnspan=9, sticky=W)

            self.tab = ttk.Notebook(self.frame)

            self.tab.bind("<<NotebookTabChanged>>", self.tabChange)

            self.tabs = [View.GeneralTab(self.tab, listener, self.tabIDCounter, self.frame, self.master)]

            for tab in self.tabs:
                self.tab.add(tab, text='Tab ' + str(self.tabIDCounter + 1))
                self.tabIDCounter += 1
            addTab = ttk.Frame(self.tab)
            self.tab.add(addTab, text='+')
            self.tabs.append(addTab)

            self.tab.grid(row=1, column=0, rowspan=9, columnspan=9, sticky='wens')

            self.frame.pack()
            self.frame.lift()

        def tabChange(self, event):
            tabIndex = event.widget.index('current')
            if len(self.tabs) > 1 and tabIndex == len(self.tabs) - 1:
                newTab = View.GeneralTab(self.tab, self.listener, self.tabIDCounter, self.frame, self.master)
                self.tab.forget(self.tabs[-1])
                self.tab.add(newTab, text='Tab ' + str(self.tabIDCounter + 1))
                self.tab.add(self.tabs[-1], text='+')
                self.tabs = self.tabs[:-1] + [newTab] + [self.tabs[-1]]
                self.tab.select(newTab)
                self.tabIDCounter += 1
                self.mMenuFile.entryconfig(2, state=tkinter.NORMAL)

        def closeTab(self):
            if len(self.tabs) != 2:
                tkId = self.tab.select()
                curTab = self.tab.nametowidget(tkId)
                curTab.close()
                ind = 0
                while self.tabs[ind] != curTab:
                    ind += 1
                self.tabs = self.tabs[:ind] + self.tabs[ind + 1:]
                if ind == len(self.tabs) - 1:
                    self.tab.select(self.tabs[-2])
                self.tab.forget(tkId)
                self.tabIDCounter = self.tabs[-2].tabID + 1
                if len(self.tabs) == 2:
                    self.mMenuFile.entryconfig(2, state=tkinter.DISABLED)

        def rechoose(self):
            tkId = self.tab.select()
            curTab = self.tab.nametowidget(tkId)
            if not curTab.listener.modelIsRunning(curTab.tabID):
                curTab.parameterFrame.destroy()
                curTab.parameterFrame = View.GeneralTab.ModelChooser(curTab)
                curTab.parameterFrame.grid(row=0, column=0, rowspan=9)
                curTab.slowLabel.grid_forget()
                curTab.slowSlider.grid_forget()
                curTab.render.grid_forget()
                curTab.displayedEpisodeNum.grid_forget()
                curTab.curEpisodeNum.grid_forget()
                curTab.graph.grid_forget()
                # curTab.graphLine.grid_forget()
                curTab.xAxisLabel.grid_forget()
                curTab.legend.grid_forget()
                # curTab.space.grid_forget()

        def train(self):
            tkId = self.tab.select()
            curTab = self.tab.nametowidget(tkId)
            # print(hasattr(curTab.parameterFrame, "train"))
            if not curTab.listener.modelIsRunning(curTab.tabID) and curTab.parameterFrame.isParameterFrame:
                curTab.parameterFrame.master.train()

        def halt(self):
            tkId = self.tab.select()
            curTab = self.tab.nametowidget(tkId)
            if curTab.parameterFrame.isParameterFrame:
                curTab.parameterFrame.master.halt()

        def test(self):
            tkId = self.tab.select()
            curTab = self.tab.nametowidget(tkId)
            if not curTab.listener.modelIsRunning(curTab.tabID) and curTab.parameterFrame.isParameterFrame:
                curTab.parameterFrame.master.test()

        def save(self):
            tkId = self.tab.select()
            curTab = self.tab.nametowidget(tkId)
            if not curTab.listener.modelIsRunning(curTab.tabID) and curTab.parameterFrame.isParameterFrame:
                curTab.parameterFrame.master.save()

        def load(self):
            tkId = self.tab.select()
            curTab = self.tab.nametowidget(tkId)
            if not curTab.listener.modelIsRunning(curTab.tabID) and curTab.parameterFrame.isParameterFrame:
                curTab.parameterFrame.master.load()

        def reset(self):
            tkId = self.tab.select()
            curTab = self.tab.nametowidget(tkId)
            if not curTab.listener.modelIsRunning(curTab.tabID) and curTab.parameterFrame.isParameterFrame:
                curTab.parameterFrame.master.reset()

        def saveResults(self):
            tkId = self.tab.select()
            curTab = self.tab.nametowidget(tkId)
            if not curTab.listener.modelIsRunning(curTab.tabID) and curTab.parameterFrame.isParameterFrame:
                curTab.parameterFrame.master.saveResults()

        def loadEnv(self):
            filename = filedialog.askopenfilename(initialdir="/", title="Select file")

            spec = importlib.util.spec_from_file_location("customenv", filename)
            try:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                tkId = self.tab.select()
                curTab = self.tab.nametowidget(tkId)
                if not curTab.listener.modelIsRunning(curTab.tabID):
                    # curTab.parameterFrame.envOpts.set(mod.CustomEnv.displayName)
                    # curTab.parameterFrame.selevUpdate()
                    # curTab.parameterFrame.slev.config(text='Selected Environment: ' + mod.CustomEnv.displayName)
                    # self.tabs[0].parameterFrame.slev
                    # self.slev.config(text='Selected Environment: ' + mod.CustomEnv.displayName)
                    View.environments = [mod.CustomEnv] + View.environments

                for ind, tab in enumerate(self.tabs):
                    if isinstance(tab, View.GeneralTab) and isinstance(tab.parameterFrame,
                                                                       View.GeneralTab.ModelChooser):
                        tab.parameterFrame.destroy()
                        tab.parameterFrame = View.GeneralTab.ModelChooser(tab)
                        tab.parameterFrame.grid(row=0, column=0, rowspan=9)
            except:
                pass

        def loadAgent(self):
            filename = filedialog.askopenfilename(initialdir="/", title="Select file")

            spec = importlib.util.spec_from_file_location("customagent", filename)
            try:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                tkId = self.tab.select()
                curTab = self.tab.nametowidget(tkId)
                if not curTab.listener.modelIsRunning(curTab.tabID):
                    # curTab.parameterFrame.agentOpts.set(mod.CustomAgent.displayName)
                    # curTab.parameterFrame.selagUpdate()
                    # curTab.parameterFrame.slag.config(text='Selected Agent: ' + mod.CustomAgent.displayName)
                    View.agents = [mod.CustomAgent] + View.agents
                #
                for ind, tab in enumerate(self.tabs):
                    if isinstance(tab, View.GeneralTab) and isinstance(tab.parameterFrame,
                                                                       View.GeneralTab.ModelChooser):
                        tab.parameterFrame.destroy()
                        tab.parameterFrame = View.GeneralTab.ModelChooser(tab)
                        tab.parameterFrame.grid(row=0, column=0, rowspan=9)
            except:
                pass

    class GeneralTab(ttk.Frame):
        def __init__(self, tab, listener, tabID, frame, master):
            super().__init__(tab)
            self.root = master
            self.frame = frame
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

            # frame = tkinter.Frame(self)
            # frame.grid(row=0, column=0, columnspan=2)
            # ttk.Label(frame, text='Number of Episodes', width=18, anchor='w').pack(side="left", padx=(5,0), pady=10)
            # ttkwidgets.tickscale.TickScale(self, from_=1, to=655360, resolution=1, orient=tkinter.HORIZONTAL)
            # self.numEpsVar = tkinter.StringVar()
            # self.numEps = ttk.Entry(frame, textvariable=self.numEpsVar).pack(side="left", padx=(195,0))
            # # numEps_ttp = View.CreateToolTip(self.numEps, "The number of episodes to run the model on")
            # self.numEpsVar.set('1000')
            #
            # frame2 = tkinter.Frame(self)
            # frame2.grid(row=1, column=0, columnspan=2)
            # ttk.Label(frame2, text='Max Steps', width=18, anchor='w').pack(side="left", padx=(5,0), pady=10)
            # ttkwidgets.tickscale.TickScale(self, from_=1, to=655360, resolution=1, orient=tkinter.HORIZONTAL)
            # self.maxStepsVar = tkinter.StringVar()
            # self.maxSteps = ttk.Entry(frame2, textvariable=self.maxStepsVar).pack(side="left", padx=(195,0))
            # # maxSteps_ttp = View.CreateToolTip(self.maxSteps, "The max number of timesteps permitted in an episode")
            # self.maxStepsVar.set('200')

            # Add model parameters here
            self.parameterFrame = self.ModelChooser(self)
            self.parameterFrame.grid(row=0, column=0, rowspan=9)
            # self.slowLabel = ttk.Label(self, text='Displayed episode speed')
            # self.slowLabel.grid(row=7, column=0)
            # self.slowSlider = ttkwidgets.tickscale.TickScale(self, from_=1, to=20, resolution=1, orient=tkinter.HORIZONTAL)
            # slowSlider_ttp = View.CreateToolTip(self.slowSlider, "The speed at which to display the episodes")
            # self.slowSlider.set(10)
            # self.slowSlider.grid(row=7, column=1)
            #
            # self.render = tkinter.Canvas(self, background='#eff0f1')
            # self.render.grid(row=0, column=2, rowspan=9, columnspan=8, sticky='wens')
            #
            # self.displayedEpisodeNum = ttk.Label(self, text='')
            # self.displayedEpisodeNum.grid(row=9, column=2)
            #
            # self.curEpisodeNum = ttk.Label(self, text='')
            # self.curEpisodeNum.grid(row=9, column=3)
            #
            # self.graph = tkinter.Canvas(self, background='#eff0f1')
            # self.graph.grid(row=10, column=2, rowspan=4, columnspan=8, sticky='wens')
            # self.graphLine = self.graph.create_line(0,0,0,0, fill='black')
            # self.graph.bind("<Motion>", self.updateGraphLine)
            # self.drawAxis()
            #
            # self.legend = tkinter.Canvas(self, background='#eff0f1')
            # self.legend.grid(row=10, column=0, rowspan=4, columnspan=2, sticky='wens')
            # self.legend.bind('<Configure>', self.legendResize)

        def legendResize(self, evt):
            self.legend.delete('all')
            h = evt.height
            p1, p2, p3, p4, p5, p6 = h / 6, 2 * h / 6, 3 * h / 6, 4 * h / 6, 5 * h / 6, 9 * h / 10
            self.legend.create_line(40, p1, 90, p1, fill='blue')
            self.legend.create_line(40, p2, 90, p2, fill='red')
            self.legend.create_line(40, p3, 90, p3, fill='green')
            self.lossLegend = self.legend.create_text(100, p1, text='MSE Episode Loss:', anchor='w')
            self.rewardLegend = self.legend.create_text(100, p2, text='Episode Reward:', anchor='w')
            self.epsilonLegend = self.legend.create_text(100, p3, text='Epsilon:', anchor='w')
            self.episodelegend = self.legend.create_text(100, p4, text='Episode:', anchor='w')
            self.testResult1 = self.legend.create_text(100, p5, text='', anchor='w')
            self.testResult2 = self.legend.create_text(100, p6, text='', anchor='w')

        def updateGraphLine(self, evt):
            xVal = evt.x
            height = self.graph.winfo_height()
            self.graph.coords(self.graphLine, [xVal, 0, xVal, height])

            if self.curTotalEpisodes:
                smoothIndex = (int)(self.curTotalEpisodes * xVal / self.graph.winfo_width()) - self.smoothAmt
                if len(self.smoothedDataPoints) > smoothIndex >= 0:
                    loss, reward, epsilon = self.smoothedDataPoints[smoothIndex]
                    self.legend.itemconfig(self.lossLegend, text='MSE Episode Loss: {:.4f}'.format(loss))
                    self.legend.itemconfig(self.rewardLegend, text='Episode Reward: ' + str(reward))
                    self.legend.itemconfig(self.epsilonLegend, text='Epsilon: {:.4f}'.format(epsilon))
                    self.legend.itemconfig(self.episodelegend, text='Episode: ' + str(smoothIndex + self.smoothAmt))

                else:
                    self.legend.itemconfig(self.lossLegend, text='MSE Episode Loss:')
                    self.legend.itemconfig(self.rewardLegend, text='Episode Reward:')
                    self.legend.itemconfig(self.epsilonLegend, text='Epsilon:')
                    self.legend.itemconfig(self.episodelegend, text='Episode:')

        def halt(self):
            self.listener.halt(self.tabID)
            self.imageQueues[0].clear()
            self.imageQueues[1].clear()
            self.imageQueuesInd = 0
            self.curImageIndDisplayed = 0
            self.isDisplayingEpisode = False
            self.waitCount = 0

        def setupRight(self):
            # self.parameterFrame.grid_forget()
            self.slowLabel = ttk.Label(self, text='Displayed episode speed')
            self.slowLabel.grid(row=4, column=1)
            self.slowSlider = ttkwidgets.tickscale.TickScale(self, from_=1, to=20, resolution=1,
                                                             orient=tkinter.HORIZONTAL)
            slowSlider_ttp = View.CreateToolTip(self.slowSlider, "The speed at which to display the episodes")
            self.slowSlider.set(10)
            self.slowSlider.grid(row=5, column=1, sticky="news")

            self.render = tkinter.Canvas(self, bg="gray80", highlightbackground="gray80")
            self.render.grid(row=4, column=2, rowspan=6, columnspan=2, sticky='wens')

            # tkinter.Canvas(self, height=15,bg="gray80").grid(row=2, column=1, rowspan=1, columnspan=1, sticky='wens')

            self.displayedEpisodeNum = ttk.Label(self, text='Showing episode')
            self.displayedEpisodeNum.grid(row=7, column=1)

            self.curEpisodeNum = ttk.Label(self, text='Episodes completed:')
            self.curEpisodeNum.grid(row=8, column=1)

            self.graph = tkinter.Canvas(self, bg="gray80", highlightbackground="gray80")
            self.graph.grid(row=0, column=2, rowspan=2, columnspan=1, sticky='wens')
            self.graphLine = self.graph.create_line(0, 0, 0, 0, fill='black')
            self.graph.bind("<Motion>", self.updateGraphLine)

            self.xAxisLabel = tkinter.Canvas(self, height=15, bg="gray80", highlightbackground="gray80")
            self.xAxisLabel.grid(row=2, column=2, rowspan=1, columnspan=1, sticky='wens')

            # self.drawAxis()
            # background='#eff0f1'
            self.legend = tkinter.Canvas(self, bg="gray80", width=275, highlightbackground="gray80")
            self.legend.grid(row=0, column=1, sticky='news')
            self.legend.bind('<Configure>', self.legendResize)

            self.space = ttk.Label(self, text=" ").grid(row=3, column=2)

            # self.columnconfigure(0, weight=1)
            # self.columnconfigure(1, weight=5)

            self.frame.pack()

        def busy(self):
            pass
            #Commented out because this crashes on Linux:
            #self.root.config(cursor="wait")

        def notbusy(self):
            self.root.config(cursor="")

        def loadingRender(self):
            self.isLoadingRender = True
            self.render.delete('all')
            w = self.render.winfo_width()
            h = self.render.winfo_height()
            self.render.create_text(w / 2, h / 2, text='Loading...', anchor='center')

        def train(self):
            self.busy()
            self.loadingRender()
            if not self.listener.modelIsRunning(self.tabID):
                self.smoothAmt = 20
                try:
                    # total_episodes = int(self.numEps.get())
                    # max_steps = int(self.maxSteps.get())

                    # self.setupPage3()

                    self.listener.startTraining(self.tabID, self.parameterFrame.getParameters())
                    self.trainingEpisodes = 0
                    self.curTotalEpisodes = self.parameterFrame.getParameters()[0]
                    self.resetGraph()
                    self.checkMessages()
                    self.legend.itemconfig(self.testResult1, text='')
                    self.legend.itemconfig(self.testResult2, text='')
                except ValueError:
                    print('Bad Hyperparameters')

        def test(self):
            self.busy()
            self.loadingRender()
            if not self.listener.modelIsRunning(self.tabID):
                self.smoothAmt = 1
                try:
                    # total_episodes = int(self.numEps.get())
                    # max_steps = int(self.maxSteps.get())

                    if not self.listener.startTesting(self.tabID, self.parameterFrame.getParameters()):
                        self.notbusy()
                        self.render.delete('all')
                        tkinter.messagebox.showerror(title="Error", message="Model has not been trained!")
                    self.trainingEpisodes = 0
                    self.curTotalEpisodes = self.parameterFrame.getParameters()[0]
                    self.resetGraph()
                    self.checkMessages()
                    self.legend.itemconfig(self.testResult1, text='')
                    self.legend.itemconfig(self.testResult2, text='')
                except ValueError:
                    print('Bad Hyperparameters')

        def save(self):
            if not self.listener.modelIsRunning(self.tabID):
                filename = filedialog.asksaveasfilename(initialdir="/", title="Select file")
                if filename:
                    self.listener.save(filename, self.tabID)

        def load(self):
            if not self.listener.modelIsRunning(self.tabID):
                filename = filedialog.askopenfilename(initialdir="/", title="Select file")
                self.listener.load(filename, self.tabID)

        def saveResults(self):
            filename = filedialog.asksaveasfilename(initialdir="/", title="Select file")
            file = open(filename, "w")

            file.write("episode, loss, reward, epsilon\n")
            for episode, (loss, reward, epsilon) in enumerate(self.graphDataPoints):
                file.write(str(episode) + "," + str(loss) + "," + str(reward) + "," + str(epsilon) + "\n")
            file.close()

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
            # self.drawAxis()
            self.graphLine = self.graph.create_line(0, 0, 0, 0, fill='black')
            self.redrawGraphXAxis()
            self.drawAxis()

        def checkMessages(self):
            if self.trainingEpisodes >= 1:
                self.notbusy()
            while self.listener.getQueue(self.tabID).qsize():
                message = self.listener.getQueue(self.tabID).get(timeout=0)
                if message.type == Model.Message.EVENT:
                    if message.data == Model.Message.EPISODE:
                        self.addEpisodeToGraph()

                        self.trainingEpisodes += 1
                        self.curEpisodeNum.configure(text='Episodes completed: ' + str(self.trainingEpisodes))
                        if self.isDisplayingEpisode:
                            self.imageQueues[self.imageQueuesInd].clear()
                        else:
                            self.imageQueuesInd = 1 - self.imageQueuesInd
                            self.imageQueues[self.imageQueuesInd].clear()
                            self.isDisplayingEpisode = True
                            self.curImageIndDisplayed = 0
                            self.displayedEpisodeNum.configure(text='Showing episode ' + str(self.trainingEpisodes))
                    elif message.data == Model.Message.TRAIN_FINISHED:
                        self.imageQueues[0].clear()
                        self.imageQueues[1].clear()
                        self.imageQueuesInd = 0
                        self.curImageIndDisplayed = 0
                        self.isDisplayingEpisode = False
                        self.waitCount = 0
                        totalReward = sum([reward for _, reward, _ in self.graphDataPoints])
                        avgReward = totalReward / len(self.graphDataPoints)
                        self.legend.itemconfig(self.testResult1, text='Total Training Reward: ' + str(totalReward))
                        self.legend.itemconfig(self.testResult2, text='Reward/Episode: ' + str(avgReward))
                        self.loadingRenderUpdate()
                        return
                    elif message.data == Model.Message.TEST_FINISHED:
                        self.imageQueues[0].clear()
                        self.imageQueues[1].clear()
                        self.imageQueuesInd = 0
                        self.curImageIndDisplayed = 0
                        self.isDisplayingEpisode = False
                        self.waitCount = 0
                        totalReward = sum([reward for _, reward, _ in self.graphDataPoints])
                        avgReward = totalReward / len(self.graphDataPoints)
                        self.legend.itemconfig(self.testResult1, text='Total Test Reward: ' + str(totalReward))
                        self.legend.itemconfig(self.testResult2, text='Reward/Episode: ' + str(avgReward))
                        self.loadingRenderUpdate()
                        return
                elif message.type == Model.Message.STATE:
                    self.imageQueues[self.imageQueuesInd].append(message.data.image)
                    self.accumulateState(message.data)

            self.updateEpisodeRender()
            self.master.after(10, self.checkMessages)

        def loadingRenderUpdate(self):
            if self.isLoadingRender:
                self.notbusy()
                self.render.delete('all')

        def addEpisodeToGraph(self):
            avgLoss = self.episodeAccLoss / self.curEpisodeSteps
            totalReward = self.episodeAccReward
            avgEpsilon = self.episodeAccEpsilon / self.curEpisodeSteps

            avgState = (avgLoss, totalReward, avgEpsilon)
            self.graphDataPoints.append(avgState)

            self.redrawGraph(len(self.graphDataPoints) % max(5, self.smoothAmt) == 0)

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
            for ind in range(0, int(self.curTotalEpisodes), step):
                x = w * (ind / self.curTotalEpisodes)
                self.graph.create_line(x, h - self.graphBottomMargin, x, h - self.graphBottomMargin / 2)
                self.graph.create_text(x, h - self.graphBottomMargin / 2, text=str(ind), anchor='n')

        def redrawGraph(self, full):
            if full:
                lastN = len(self.graphDataPoints)
                self.curLossAccum = 0
                self.curRewardAccum = 0
                self.smoothedDataPoints.clear()
                self.lossGraphMax = max(0.0000000000001, sorted([loss for loss, _, _ in self.graphDataPoints])[
                    int((len(self.graphDataPoints) - 1) * 0.95)] * 1.1)
                rewardSorted = sorted([reward for _, reward, _ in self.graphDataPoints])
                self.rewardGraphMax = rewardSorted[int((len(self.graphDataPoints) - 1) * 0.95)]
                self.rewardGraphMin = rewardSorted[int((len(self.graphDataPoints) - 1) * 0.05)]
                extendAmt = 0.1 * (self.rewardGraphMax - self.rewardGraphMin)
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
            for ind in range(max(0, offset), len(self.graphDataPoints)):
                oldX = w * (ind / self.curTotalEpisodes)
                newX = w * ((ind + 1) / self.curTotalEpisodes)
                avgLoss, totalReward, avgEpsilon = self.graphDataPoints[ind]
                if ind > 0:
                    _, _, prevEpsilon = self.graphDataPoints[ind - 1]
                    oldY = h * (1 - prevEpsilon)
                    newY = h * (1 - avgEpsilon)
                    self.graph.create_line(oldX, oldY, newX, newY, fill='green')

                if ind >= self.smoothAmt:
                    prevLoss, prevReward = self.curLossAccum / self.smoothAmt, self.curRewardAccum / self.smoothAmt
                    (obsLoss, obsReward, _) = self.graphDataPoints[ind - self.smoothAmt]

                    self.curLossAccum -= obsLoss
                    self.curRewardAccum -= obsReward
                    self.curLossAccum += avgLoss
                    self.curRewardAccum += totalReward

                    curReward = self.curRewardAccum / self.smoothAmt
                    curLoss = self.curLossAccum / self.smoothAmt
                    self.smoothedDataPoints.append((curLoss, curReward, avgEpsilon))

                    rewardRange = max(0.000000001, self.rewardGraphMax - self.rewardGraphMin)
                    oldY = (self.graphBottomMargin + (h - self.graphBottomMargin) * (
                            1 - (prevReward - self.rewardGraphMin) / rewardRange)) - 4
                    newY = (self.graphBottomMargin + (h - self.graphBottomMargin) * (
                            1 - (curReward - self.rewardGraphMin) / rewardRange)) - 4
                    self.graph.create_line(oldX, oldY, newX, newY, fill='red')

                    oldY = h * (1 - prevLoss / self.lossGraphMax)
                    newY = h * (1 - curLoss / self.lossGraphMax)
                    self.graph.create_line(oldX, oldY, newX, newY, fill='blue')
                else:
                    self.curLossAccum += avgLoss
                    self.curRewardAccum += totalReward
            self.drawAxis()

        def drawAxis(self):
            self.graph.create_line(2, 0, 2, self.graph.winfo_height(), fill='black')
            # self.graph.create_line(0, int(self.graph.winfo_height()/2), self.graph.winfo_width(),
            #                        int(self.graph.winfo_height()/2), fill='black')
            self.graph.create_line(0, self.graph.winfo_height() - 3, self.graph.winfo_width(),
                                   self.graph.winfo_height() - 3, fill='black')
            self.xAxisLabel.create_text(int(self.xAxisLabel.winfo_width() / 2), int(self.xAxisLabel.winfo_height() / 2),
                                        text='Timestamp', anchor='center')

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
                    self.curImageIndDisplayed = self.curImageIndDisplayed + 1
                    if self.curImageIndDisplayed == len(displayQueue):
                        self.curImageIndDisplayed = 0
                        self.isDisplayingEpisode = False
                    if tempImage:
                        tempImage = tempImage.resize((self.render.winfo_width(), self.render.winfo_height()))
                        self.image = ImageTk.PhotoImage(
                        tempImage)  # must maintain a reference to this image in self: otherwise will be garbage collected
                        if self.renderImage:
                            self.render.delete(self.renderImage)
                            self.isLoadingRender = False
                        self.renderImage = self.render.create_image(0, 0, anchor='nw', image=self.image)
                self.waitCount += 1

        def selectModel(self):
            agent, env = None, None
            for curAgent in View.agents:
                if self.parameterFrame.agentOpts.get() == curAgent.displayName:
                    agent = curAgent
                    break
            for curEnv in View.environments:
                if self.parameterFrame.envOpts.get() == curEnv.displayName:
                    env = curEnv
                    break

            if agent and env:
                self.master.tab(self, text=agent.displayName + '+' + env.displayName)
                self.parameterFrame.destroy()
                self.parameterFrame = self.ParameterFrame(self, agent, env)
                self.parameterFrame.grid(row=0, column=0, rowspan=9)
                self.setupRight()
            else:
                messagebox.showerror("Error", "Please select both an agent and an environment")

        def close(self):
            self.listener.close(self.tabID)

        class ParameterFrame(ttk.Frame):
            def __init__(self, master, agentClass, envClass):
                super().__init__(master)
                self.isParameterFrame = True
                self.master = master
                master.listener.setAgent(master.tabID, agentClass)
                master.listener.setEnvironment(master.tabID, envClass)
                self.values = []

                self.createParameterChooser(
                    agent.Agent.Parameter('Number of Episodes', 1, 655360, 1, 1000, True, True,
                                          "The number of episodes to run the model on"))
                self.createParameterChooser(
                    agent.Agent.Parameter('Max Size', 1, 655360, 1, 200, True, True,
                                          "The max number of timesteps permitted in an episode"))
                for param in agentClass.parameters:
                    self.createParameterChooser(param)

                # train = ttk.Button(self, text='Train', command=self.master.train)
                # train.pack(side='left')
                # train_button_ttp = View.CreateToolTip(train, "Train the agent with the current settings")
                # halt = ttk.Button(self, text='Halt', command=self.master.halt)
                # halt.pack(side='left')
                # halt_button_ttp = View.CreateToolTip(halt, "Pause the current training")
                # test = ttk.Button(self, text='Test', command=self.master.test)
                # test.pack(side='left')
                # test_button_ttp = View.CreateToolTip(test, "Test the agent in its current state")
                # save = ttk.Button(self, text='Save Agent', command=self.master.save)
                # save.pack(side='left')
                # save_button_ttp = View.CreateToolTip(save, "Save the agent in its current state")
                # load = ttk.Button(self, text='Load Agent', command=self.master.load)
                # load.pack(side='left')
                # load_button_ttp = View.CreateToolTip(load, "Load an agent")
                # reset = ttk.Button(self, text='Reset', command=self.master.reset)
                # reset.pack(side='left')
                # reset_button_ttp = View.CreateToolTip(reset, "Reset the current agent and its parameters")
                # save_results = ttk.Button(self, text='Save Results', command=self.master.saveResults)
                # save_results.pack(side='left')
                # save_results_button_ttp = View.CreateToolTip(save_results, "Save the results of the current training session")

            def createParameterChooser(self, param):
                subFrame = ttk.Frame(self)
                ttk.Label(subFrame, text=param.name, width=18).pack(side="left", expand=True, fill='both', padx=5,
                                                                    pady=5)
                valVar = tkinter.StringVar()
                input = None

                def scaleChanged(val):
                    if subFrame.focus_get() != input:
                        valVar.set(val)

                scale = ttkwidgets.tickscale.TickScale(subFrame, from_=param.min, to=param.max,
                                                       resolution=param.resolution,
                                                       orient=tkinter.HORIZONTAL, command=scaleChanged, length=170)
                View.CreateToolTip(scale, param.toolTipText)

                scale.set(param.default)
                scale.pack(side="left", expand=True, fill='both', padx=5, pady=5)

                def entryChanged(var, indx, mode):
                    try:
                        if subFrame.focus_get() == input:
                            scale.set(float(valVar.get()))
                    except ValueError:
                        pass

                valVar.trace_add('write', entryChanged)
                input = ttk.Entry(subFrame, textvariable=valVar)
                valVar.set(str(param.default))
                input.pack(side="right", expand=True, padx=5, pady=5)
                subFrame.pack(side='top')
                self.values.append(scale)

            def getParameters(self):
                return [value.get() for value in self.values]

        class ModelChooser(ttk.Frame):
            def __init__(self, master):
                super().__init__(master)
                self.isParameterFrame = False
                self.agentOpts = tkinter.StringVar(self)
                self.envOpts = tkinter.StringVar(self)
                self.envButtons = []
                self.agentButtons = []
                subFrame = ttk.Frame(self)

                envName = [opt.displayName for opt in View.environments]
                agtName = [opt.displayName for opt in View.agents]

                # ttk.Combobox(subFrame, state='readonly', values=agtName, textvariable = self.agentOpts).pack(side='left')
                # ttk.Combobox(subFrame, state='readonly', values=envName, textvariable = self.envOpts).pack(side='left')

                imgloc = "./img/"
                imty = '.jpg'

                entxb = tkinter.Text(subFrame, height=5, width=137, wrap=tkinter.NONE, bg="gray80")
                enscb = ttk.Scrollbar(subFrame, orient=tkinter.HORIZONTAL, command=entxb.xview)
                entxb.configure(xscrollcommand=enscb.set)
                enscb.pack(fill=tkinter.X)
                entxb.pack()
                self.slev = ttk.Label(subFrame, text='Selected Environment: None')
                self.slev.pack(pady=(15, 75))
                # style = Style()
                # style.configure('TButton', activebackground="gray80",
                #                 borderwidth='4', )

                for e in envName:
                    try:
                        epic = Image.open(imgloc + e + imty)
                        epic = epic.resize((50, 50), Image.ANTIALIAS)
                        piepic = PhotoImage(epic)

                        eb = tkinter.Radiobutton(entxb, image=piepic, text=e, variable=self.envOpts, value=e,
                                                 command=self.selevUpdate, compound=tkinter.TOP, indicatoron=0,
                                                 height=70)
                        eb.piepic = piepic
                        self.envButtons.append(eb)
                    except IOError:
                        epic = Image.open(imgloc + "custom" + imty)
                        epic = epic.resize((50, 50), Image.ANTIALIAS)
                        piepic = PhotoImage(epic)

                        eb = tkinter.Radiobutton(entxb, image=piepic, text=e, variable=self.envOpts, value=e,
                                                 command=self.selevUpdate, compound=tkinter.TOP, indicatoron=0,
                                                 height=70)
                        eb.piepic = piepic
                        self.envButtons.append(eb)
                    #     anchor=tkinter.S
                    entxb.window_create(tkinter.END, window=eb)

                entxb.configure(state=tkinter.DISABLED)

                agtxb = tkinter.Text(subFrame, height=2, width=137, wrap=tkinter.NONE, bg="gray80")
                agscb = ttk.Scrollbar(subFrame, orient=tkinter.HORIZONTAL, command=agtxb.xview)
                agtxb.configure(xscrollcommand=agscb.set)
                agscb.pack(fill=tkinter.X)
                agtxb.pack()
                self.slag = ttk.Label(subFrame, text='Selected Agent: None')
                self.slag.pack(pady=(15, 30))

                for a in agtName:
                    ab = tkinter.Radiobutton(agtxb, text=a, variable=self.agentOpts, value=a, command=self.selagUpdate,
                                             compound=tkinter.TOP, indicatoron=0, height=1)
                    agtxb.window_create(tkinter.END, window=ab)
                    self.agentButtons.append(ab)

                agtxb.configure(state=tkinter.DISABLED)

                subFrame.pack()
                set_model = tkinter.Button(self, text='Set Model', command=master.selectModel)
                set_model.pack()

                space = tkinter.Canvas(self, bg="gray80", highlightbackground="gray80")
                space.pack()
                View.CreateToolTip(set_model, "Run program with the currently selected environment and agent")

            def selevUpdate(self):
                envUpdate = 'Selected Environment: ' + self.envOpts.get()
                self.slev.config(text=envUpdate)
                curAgents = View.allowedAgents.get(self.envOpts.get())
                if curAgents is not None:
                    for agentButton in self.agentButtons:
                        if agentButton.cget('text') in curAgents or agentButton.cget('text') not in View.allowedEnvs:
                            agentButton.configure(state=tkinter.NORMAL)
                        else:
                            agentButton.configure(state=tkinter.DISABLED)
                            if agentButton.cget('text') == self.agentOpts.get():
                                self.agentOpts.set(None)

                else:
                    for agentButton in self.agentButtons:
                        agentButton.configure(state=tkinter.NORMAL)
                for envButton in self.envButtons:
                    envButton.configure(state=tkinter.NORMAL)

            def selagUpdate(self):
                agUpdate = 'Selected Agent: ' + self.agentOpts.get()
                self.slag.config(text=agUpdate)
                curEnvs = View.allowedEnvs.get(self.agentOpts.get())
                if curEnvs is not None:
                    for envButton in self.envButtons:
                        if envButton.cget('text') in curEnvs or envButton.cget('text') not in View.allowedAgents:
                            envButton.configure(state=tkinter.NORMAL)
                        else:
                            envButton.configure(state=tkinter.DISABLED)
                            if envButton.cget('text') == self.envOpts.get():
                                self.envOpts.set(None)
                else:
                    for envButton in self.envButtons:
                        envButton.configure(state=tkinter.NORMAL)
                for agentButton in self.agentButtons:
                    agentButton.configure(state=tkinter.NORMAL)

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


def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
