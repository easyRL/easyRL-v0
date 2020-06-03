import tkinter

from MVC import view, model
from ttkthemes import ThemedTk
import threading
import queue
from tkinter import Menu, ttk

# pip install pillow
# pip install gym
# pip install pandas
# pip install numpy
# pip install tensorflow
# pip install opencv-python
# pip install gym[atari]  (if not on Windows)
# OR if on Windows:
# pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
# pip install git+https://github.com/Kojoley/atari-py.git
# pip install ttkthemes
# pip install ttkwidgets

class Controller:
    def __init__(self):
        self.models = {}
        self.viewListener = self.ViewListener(self)
        self.root = ThemedTk(theme='breeze')
        self.view = view.View(self.root, self.viewListener)

        def helpMenu():
            popup = tkinter.Tk()
            popup.wm_title("Help")

            texts = "To train a model choose New Agent.\nYou may then select your agent and environment through the drop down menus and choose Set Model.\n"
            texts += "You can use the sliders to set the hyperparameters for training your agent.\n"
            texts += "Press Train to begin training the agent, you may stop the training with Halt or change the playback speed with the episode speed slider."
            label = ttk.Label(popup, text=texts)
            label.pack(side="top", fill="x", pady=10)
            B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
            B1.pack()
            popup.mainloop()

        self.menubar = Menu(self.root)
        self.menubar.add_command(label="Help", command=helpMenu)
        self.root.config(menu=self.menubar)
        
        self.root.protocol("WM_DELETE_WINDOW", self.delete_window)
        self.root.mainloop()

    class ViewListener:
        def __init__(self, controller):
            self.controller = controller
            self.messageQueues = {}

        def getModel(self, tabID):
            curModel = self.controller.models.get(tabID)
            if not curModel:
                curModel = model.Model()
                self.controller.models[tabID] = curModel
            return curModel

        def getQueue(self, tabID):
            curQueue = self.messageQueues.get(tabID)
            if not curQueue:
                curQueue = queue.Queue()
                self.messageQueues[tabID] = curQueue
            return curQueue

        def setEnvironment(self, tabID, envClass):
            model = self.getModel(tabID)
            model.reset()
            model.environment_class = envClass
            print('loaded ' + envClass.displayName)

        def setAgent(self, tabID, agentClass):
            model = self.getModel(tabID)
            model.reset()
            model.agent_class = agentClass
            print('loaded ' + agentClass.displayName)

        def startTraining(self, tabID, args):
            model = self.getModel(tabID)
            queue = self.getQueue(tabID)
            threading.Thread(target=model.run_learning, args=[queue,]+args).start()

        def startTesting(self, tabID, args):
            model = self.getModel(tabID)
            queue = self.getQueue(tabID)
            threading.Thread(target=model.run_testing, args=[queue,]+args).start()

        def modelIsRunning(self, tabID):
            model = self.getModel(tabID)
            return model.isRunning

        def halt(self, tabID):
            model = self.getModel(tabID)
            model.halt_learning()

        def reset(self, tabID):
            model = self.getModel(tabID)
            model.reset()

        def close(self, tabID):
            self.halt(tabID)
            if self.controller.models.get(tabID):
                del self.controller.models[tabID]
            if self.messageQueues.get(tabID):
                del self.messageQueues[tabID]

        def save(self, filename, tabID):
            model = self.getModel(tabID)
            model.save(filename)

        def load(self, filename, tabID):
            model = self.getModel(tabID)
            model.load(filename)

    def delete_window(self):
        for _, model in self.models.items():
            model.halt_learning()
        try:
            self.root.destroy()
        except:
            pass

# Conventional way to write the main method
if __name__ == "__main__":
    Controller()

