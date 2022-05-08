#https://stackoverflow.com/questions/23599087/multiprocessing-python-core-foundation-error/23982497

from multiprocessing import Process, Queue
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def tkapp_thread(q):
    import tkinter as tk
    from tkapp_thread import Application
    root = tk.Tk()
    app = Application(root, q, video_source=0)
    app.mainloop()
    
def yolov5_thread(q):
    from yolov5_detect import yolov5_detection
    from option_parser import get_parser
    opt = get_parser()
    yolov5_detection(q, opt)
    
if __name__ == '__main__':
    q = Queue()
    p1 = Process(target = tkapp_thread, args=(q,))
    p2 = Process(target = yolov5_thread, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
