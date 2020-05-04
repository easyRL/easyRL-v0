import tkinter
from tkinter import ttk

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
            self.listener.setEnvironment('frozenlake')
            View.ProjectWindow(self.master, self.listener)
            self.frame.destroy()

        def chooseCartPole(self):
            self.listener.setEnvironment('cartpole')
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

            self.qlearningTab = tkinter.Frame(self.tab)
            self.deepQTab = tkinter.Frame(self.tab)
            self.deepSarsaTab = tkinter.Frame(self.tab)
            self.tab.add(self.qlearningTab, text='Q Learning')
            self.tab.add(self.deepQTab, text='Deep Q Learning')
            self.tab.add(self.deepSarsaTab, text='Deep SARSA')

            self.tab.grid(row=1, column=0, rowspan=9, columnspan=10, sticky='wens')

            self.frame.grid(row=0, column=0)
            self.frame.lift()

    class GraphicsArea:
        def __init__(self, view):
            self.width = 800
            self.height = 400
            self.canvas = tkinter.Canvas(view.frame, width=self.width, height=self.height)
            self.canvas.grid(row=0, column=0, columnspan=10)
            self.backgroundId = self.canvas.create_rectangle(0, 0, self.width, self.height, fill='white')
            self.canvas.tag_bind(self.backgroundId, '<ButtonPress-1>', self.canvasClick)
            self.nodeMap = {}
            self.origin = None
            self.clickFlag = 'none'

        def canvasClick(self, event):
            pass