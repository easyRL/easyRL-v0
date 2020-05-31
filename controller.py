import view
import model
import tkinter
import cartPoleEnv
import cartPoleEnvDiscrete
import frozenLakeEnv
import qLearning
import deepQ
import threading
import queue

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

class Controller:
    def __init__(self):
        self.models = {}
        self.viewListener = self.ViewListener(self)
        self.root = tkinter.Tk(className='rl framework')
        self.view = view.View(self.root, self.viewListener)
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
            self.getModel(tabID).environment_class = envClass
            print('loaded ' + envClass.displayName)

        def setAgent(self, tabID, agentClass):
            self.getModel(tabID).agent_class = agentClass
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
            model = self.getMode(tabID)
            model.agent.reset()

        def close(self, tabID):
            self.halt(tabID)
            if self.controller.models.get(tabID):
                del self.controller.models[tabID]
            if self.messageQueues.get(tabID):
                del self.messageQueues[tabID]

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

