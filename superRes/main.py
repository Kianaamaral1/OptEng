# !pip install --upgrade opencv-python
# !pip install --upgrade opencv-contrib-python
import cv2
from cv2 import dnn_superres

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
image = cv2.imread('input.png')

# Read the desired model
path = './model/EDSR_x4.pb'
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 4)

# Upscale the image
result = sr.upsample(image)

# Save the image
cv2.imwrite("./edsrx4.png", result)
