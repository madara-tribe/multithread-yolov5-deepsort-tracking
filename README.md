# tk-yolov5-tracking
## Versions
- python 3.8.1
- pytorch 1.9.1+cu102
- tkinter 8.6.10

# abstract: multi-thread with yolov5-thread and tkinter-thread

working yolov5-thread and tkinter-thread as multithread on Mac. overall stracture is bellow.
<img src="https://user-images.githubusercontent.com/48679574/135762367-b2045111-4764-4518-90b2-6fc668f039a4.png" width="600px">

## yolov5 pretrained weight
- [yolov5 pretrained weight](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiMn431za7zAhU0yosBHVh9CbEQFnoECAIQAQ&url=https%3A%2F%2Fgithub.com%2Fultralytics%2Fyolov5%2Freleases%2Fdownload%2Fv4.0%2Fyolov5s.pt&usg=AOvVaw2MEXDNRJv1NASj6H-3qBQY)

## dataset 

- [traffic.mp4](https://drive.google.com/file/d/1UTwwrF8YRTqOS4RnkBvHip7R25M0avFo/view?usp=sharing)
- [messi_pk.mp4](https://drive.google.com/file/d/1LPOAAtgZFOQ5FQWOwtM3mcWrmeb3GULH/view?usp=sharing)
- [live_camera.mp4](https://drive.google.com/file/d/1hE75k0HT7s8Nxsxh9ip78Q9Xx5bjeyeN/view?usp=sharing)


# Deepsort tracking

DeepSort is popular tracking algrism, often using with object detection.


<img src="https://user-images.githubusercontent.com/48679574/135762464-b10bb172-7364-484f-bb7b-3f65fbf55dcc.jpeg" width="500px">


# how to start

if you just detct by yolov5. it can move in Docker.
```
$ python3 yolov5_detect.py
```

if you move mutithread. it can be moved on like Mac, because it use tkinter.
```
$ python3 multithread_tkapp.py 
```

# demo script : multithread with yolov5 and tkinter
```python3
from multiprocessing import Process, Queue

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
    yolov5_detection(q, opt, save_vid=False, show_vid=False, tkinter_is=True)
    
if __name__ == '__main__':
    q = Queue()
    p1 = Process(target = tkapp_thread, args=(q,))
    p2 = Process(target = yolov5_thread, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

```

# output

## movie1

![trffic](https://user-images.githubusercontent.com/48679574/135762475-c6c995ac-b72e-4474-94b0-872085a35b0b.gif)

## movie2

![webcam](https://user-images.githubusercontent.com/48679574/135762477-6c4361d7-9289-4784-923f-ce2a1027383a.gif)



# References
- [Deepsort-pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
- [yolov5](https://github.com/ultralytics/yolov5)

