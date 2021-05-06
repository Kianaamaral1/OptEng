#This program applies a super resolution model to an image to produce a higher resolution image.
#4/15/21 BIOL 667 Optical Engineering Spring 2021
#Battle of the Scopes: Kiana Amaral, Faven Berhane, Tianchen Liu, Cara Ly, Alennie Roldan

#V1 3/16/21 Main program created
#V2 4/15/21 Reformatting changes, added additional comments, program description, and program history

###LIBRARIES###
# !pip install --upgrade opencv-python
# !pip install --upgrade opencv-contrib-python
import cv2
from cv2 import dnn_superres

###USER SETTINGS###
image = cv2.imread('cheek2_Alennie.jpeg')      # Read image; USER: change name according to image file names
path = './model/LapSRN_x4.pb'                 # Set path to model; USER: change model name according to desired model

###MAIN PROGRAM###
sr = dnn_superres.DnnSuperResImpl_create()  # Create an SR object

sr.readModel(path)      # Read the desired model

sr.setModel("lapsrn", 4)  # USER: Set the desired model and scale to get correct pre- and post-processing

result = sr.upsample(image) # Upscale the image

cv2.imwrite("./lapsrnx4_cheekV2.png", result)  #  USER: Save the image



