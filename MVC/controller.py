import tkinter

from MVC import view, model
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
# {
    # pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
    # pip install git+https://github.com/Kojoley/atari-py.git
# }
# pip install ttkthemes
# pip install ttkwidgets

class Controller:
    def __init__(self):
        self.models = {}
        self.viewListener = self.ViewListener(self)
        self.view = view.View(self.viewListener)
        #self.view = terminalView.View(self.viewListener)

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

        def haltAll(self):
            for _, model in self.controller.models.items():
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

# Conventional way to write the main method
if __name__ == "__main__":
    Controller()

