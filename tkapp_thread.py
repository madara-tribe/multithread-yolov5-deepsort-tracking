import tkinter as tk
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
from tkinter import font
import time
from multiprocessing import Queue


class Application(tk.Frame):
    def __init__(self,master, q:Queue, video_source=0):
        super().__init__(master)

        self.master.geometry("1280x768")
        self.master.title("Tkinter with Video Streaming and Capture")
        
        self.q = q
        
        self.font_setup()
        self.vcap = cv2.VideoCapture( video_source )
        self.width = self.vcap.get( cv2.CAP_PROP_FRAME_WIDTH )
        self.height = self.vcap.get( cv2.CAP_PROP_FRAME_HEIGHT )

        self.create_widgets()
        self.create_frame_button(self.master)
        self.delay = 15 #[ms]
        self.update()


    def font_setup(self):
        self.font_frame = font.Font( family="Meiryo UI", size=15, weight="normal" )
        self.font_btn_big = font.Font( family="Meiryo UI", size=20, weight="bold" )
        self.font_btn_small = font.Font( family="Meiryo UI", size=15, weight="bold" )

        self.font_lbl_bigger = font.Font( family="Meiryo UI", size=45, weight="bold" )
        self.font_lbl_big = font.Font( family="Meiryo UI", size=30, weight="bold" )
        self.font_lbl_middle = font.Font( family="Meiryo UI", size=15, weight="bold" )
        self.font_lbl_small = font.Font( family="Meiryo UI", size=12, weight="normal" )
        
    def create_widgets(self):

        #Frame_Camera
        self.frame_cam = tk.LabelFrame(self.master, text = 'Camera', font=self.font_frame)
        self.frame_cam.place(x = 10, y = 10)
        self.frame_cam.configure(width = self.width+30, height = self.height+50)
        self.frame_cam.grid_propagate(0)

        #Canvas
        self.canvas1 = tk.Canvas(self.frame_cam)
        self.canvas1.configure( width= self.width, height=self.height)
        self.canvas1.grid(column= 0, row=0,padx = 10, pady=10)

    def create_frame_button(self, root):
        # Frame_Button
        self.frame_btn = tk.LabelFrame(root, text='Control', font=self.font_frame)
        self.frame_btn.place(x=10, y=650 )
        self.frame_btn.configure(width=self.width, height=120 )
        self.frame_btn.grid_propagate(0)

        # Close
        self.btn_close = tk.Button( self.frame_btn, text='Close', font=self.font_btn_big )
        self.btn_close.configure( width=15, height=1, command=self.press_close_button )
        self.btn_close.grid( column=1, row=0, padx=20, pady=10 )


    def update(self):
        #Get a frame from the video source
        frame = self.q.get() #cv2.imread(self.yolov5_img_path)
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))

        #self.photo -> Canvas
        self.canvas1.create_image(0,0, image= self.photo, anchor = tk.NW)
        self.master.after(self.delay, self.update)

    def press_close_button(self):
        self.master.destroy()
        self.vcap.release()
        self.canvas.delete("o")
#if __name__ == "__main__":
 #   root = tk.Tk()
  #  app = Application(master=root)#Inherit
   # app.mainloop()
