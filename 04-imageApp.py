from tkinter import *
from PIL import Image, ImageTk

canvas_width = 270
canvas_height = 270

master = Tk()
master.minsize(200,150)

canvas = Canvas(master, width=canvas_width, height=canvas_height, bg = 'white')
canvas.pack()

image = "emojis/angry.png"

img = ImageTk.PhotoImage(Image.open(image))
canvas.create_image(40, 40, anchor=NW, image=img)

master.mainloop()