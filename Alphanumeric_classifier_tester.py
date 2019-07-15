from tkinter import *

import matplotlib.pyplot as plt
import numpy as np

import PIL
from PIL import Image, ImageDraw, ImageTk


class ACP(object):
    def __init__(self,image_canvas_shape,prediction_canvas_shape,button_canvas_shape,root_widget,model,mapping_dict):
        self.image_canvas = Canvas(root_widget,width = image_canvas_shape[0],height = image_canvas_shape[1],bg = "black")
        self.image_canvas.pack()
        self.image_canvas.bind("<B1-Motion>",self.paint)

        self.prediction_canvas = Canvas(root_widget,width = prediction_canvas_shape[0],height = prediction_canvas_shape[1],bg = "snow")
        self.prediction_canvas.pack()

        self.button_canvas = Canvas(root_widget,height = button_canvas_shape[0],width = button_canvas_shape[1],bg = "snow")
        self.button_canvas.pack()

        self.image_object = PIL.Image.new("L",image_canvas_shape,color = 0)
        self.draw_object = ImageDraw.Draw(self.image_object)
        self.model = model
        self.mapping_dict = mapping_dict

        button_1 = Button(master = self.button_canvas,text = "Clear",command = self.clear)
        button_2 = Button(master = self.button_canvas,text = "Predict",command = self.predict)

        button_1.pack()
        button_2.pack()

        root_widget.mainloop()
        
    def paint(self,event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.image_canvas.create_line(x1, y1, x2, y2, fill = "snow" ,width=5)
        self.draw_object.line([x1, y1, x2, y2],fill = 255,width=5)

    def clear(self):
        self.image_canvas.delete("all")
        self.prediction_canvas.delete("all")
        self.image_object = PIL.Image.new("L",self.image_object.size,color = 0)
        self.draw_object = ImageDraw.Draw(self.image_object)

    def predict(self):
        model_input_shape = self.model.layers[0].get_input_at(0).get_shape()

        self.prediction_canvas.delete("all")
        
        image_resized = self.image_object.resize(model_input_shape[1:-1],PIL.Image.ANTIALIAS)
        image_resized.save("image.png")

        input_list = list(model_input_shape[1:])
        input_list.insert(0,1)

        image_resized = np.asarray(image_resized).reshape(tuple(input_list)) / 255

        prediction = self.model.predict(image_resized)
        
        self.prediction_canvas.create_text(int(self.prediction_canvas["width"]) / 2,int(self.prediction_canvas["height"]) / 2,text = "The network predictes: " + self.mapping_dict[str(np.argmax(prediction[0]))])
