import view
import model
import tkinter

class Controller:
    def __init__(self):
        self.model = model.Model()
        self.viewListener = ViewListener(self)
        root = tkinter.Tk(className='rl framework')
        self.view = view.View(root, self.viewListener)
        root.mainloop()

class ViewListener:
    def __init__(self, controller):
        self.controller = controller

    def callback1(self):
        print("callback1 called")

    def callback2(self):
        print("callback2 called")

def main():
    Controller()

main()