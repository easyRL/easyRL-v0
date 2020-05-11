import view
import model
import tkinter
import cartPoleEnv
import frozenLakeEnv
import qLearning
import threading
import queue

# pip install pillow
# pip install gym
# pip install pandas
# pip install numpy

class Controller:
    def __init__(self):
        self.model = model.Model()
        self.viewListener = self.ViewListener(self)
        self.root = tkinter.Tk(className='rl framework')
        self.view = view.View(self.root, self.viewListener)
        self.root.protocol("WM_DELETE_WINDOW", self.delete_window)
        self.root.mainloop()

    class ViewListener:
        def __init__(self, controller):
            self.controller = controller
            self.messageQueue = queue.Queue()

        def setFrozenLakeEnv(self):
            self.controller.model.environment = frozenLakeEnv.FrozenLakeEnv()
            print('loaded frozen lake')

        def setCartPoleEnv(self):
            self.controller.model.environment = cartPoleEnv.CartPoleEnv()
            print('loaded cartpole')

        def setQLearningAgent(self):
            self.controller.model.agent_class = qLearning.QLearning

        def setDeepQLearningAgent(self):
            pass

        def setDeepSarsaAgent(self):
            pass

        def startTraining(self, *args):
            threading.Thread(target=self.controller.model.run_learning, args=(self.messageQueue,)+args).start()

        def modelIsRunning(self):
            return self.controller.model.isRunning

        def halt(self):
            self.controller.model.halt_learning()

        def reset(self):
            if self.controller.model.agent.reset:
                self.controller.model.agent.reset()
    def delete_window(self):
        self.viewListener.halt()
        try:
            self.root.destroy()
        except:
            pass

# Conventional way to write the main method
if __name__ == "__main__":
    Controller()

