from tkinter import *

from keras.models import load_model

from Alphanumeric_classifier_tester import ACP

model_file_name = "MNIST_final.h5"
mapping_file_name = "MNIST_mapping.txt"

model = load_model(model_file_name)

root = Tk()
image_canvas_shape = (300,300)
prediction_canvas_shape = (300,50)
button_canvas_shape = (100,100)

with open(mapping_file_name) as file:
    model_mapping = eval(file.read())

app = ACP(image_canvas_shape,prediction_canvas_shape,button_canvas_shape,root,model,model_mapping)