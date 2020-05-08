import tkinter
from tkinter import ttk
from PIL import Image, ImageTk

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
            self.tab.add(self.deepSarsaTab, text='Deep SARSA')

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
            self.decayRate = tkinter.Scale(self, from_=0.00, to=1, resolution=0.01, orient=tkinter.HORIZONTAL)
            self.decayRate.grid(row=6, column=1)

            self.trainButton = tkinter.Button(self, text='Train', fg='black', command=self.train)
            self.trainButton.grid(row=7, column=0)

            self.canvas = tkinter.Canvas(self)
            self.canvas.grid(row=0, column=2, rowspan=9, columnspan=8, sticky='wens')

        def train(self):
            try:
                total_episodes = int(self.numEps.get())
                learning_rate = self.learningRate.get()
                max_steps = int(self.maxSteps.get())
                gamma = self.gamma.get()
                max_epsilon = self.maxEpsilon.get()
                min_epsilon = self.minEpsilon.get()
                decay_rate = self.decayRate.get()

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
            message = None
            finished = False
            while self.listener.messageQueue.qsize():
                message = self.listener.messageQueue.get(timeout=0)
            if message:
                if message.image:
                    tempImage = message.image.resize((self.canvas.winfo_width(), self.canvas.winfo_height()))
                    self.image = ImageTk.PhotoImage(tempImage)
                    self.canvas.create_image(0, 0, anchor='nw', image=self.image)
                if message.done:
                    finished = True
            if not finished:
                self.master.after(20, self.checkMessages)


    class GraphicsArea:
        def __init__(self, view):
            self.canvas = tkinter.Canvas(view.frame, width=self.width, height=self.height)
            self.canvas.grid(row=0, column=0, columnspan=10)
            self.backgroundId = self.canvas.create_rectangle(0, 0, self.width, self.height, fill='white')
            self.canvas.tag_bind(self.backgroundId, '<ButtonPress-1>', self.canvasClick)
            self.nodeMap = {}
            self.origin = None
            self.clickFlag = 'none'

        def canvasClick(self, event):
            pass