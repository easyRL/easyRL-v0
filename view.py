import tkinter

class View:
    def __init__(self, master, listener):
        self.frame = tkinter.Frame(master)
        self.listener = listener
        self.levelCreate = ViewArea(self.frame)
        self.frame.grid(row=0, column=0)

        self.button = tkinter.Button(self.frame, text='button', fg='red', command=self.handleButton)
        self.button.grid(row=1, column=0)

    def handleButton(self):
        self.listener.callback1()

class ViewArea:
    def __init__(self, master):
        self.width = 800
        self.height = 400
        self.canvas = tkinter.Canvas(master, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0, columnspan=10)
        self.backgroundId = self.canvas.create_rectangle(0, 0, self.width, self.height, fill='white')
        self.canvas.tag_bind(self.backgroundId, '<ButtonPress-1>', self.canvasClick)
        self.nodeMap = {}
        self.origin = None
        self.clickFlag = 'none'

    def canvasClick(self, event):
        pass