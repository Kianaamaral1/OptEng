"""
Reconstruct one image using + and - keys
Press 's' to save reconstructed image with format "foo_120.jpg" where foo is the input file name, 120 is the reconstruction Z

V1 4.8.21 image format= did_13_3940.jpg where 13 is image number, Z in um, if z=0, no focus is provided
Tom Zimmerman, CCC, IBM Research
This work is funded by the National Science Foundation (NSF) grant No. DBI-1548297 
Disclaimer:  Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.
"""
import numpy as np
import cv2

######## PUT YOUR FILE NAME HERE ###########
imageFile = 'cheek5.jpeg'
############################################

################## SETTINGS ################
defaultZ=2000 # starting z value
zStep = 10 # how many Z values to traverse at a time.
dxy   = 1.4e-6 # imager pixel size in meters.
wvlen = 650.0e-9 # Red 
zScale=1e-6 # convert z units to microns 
DISPLAY_REZ=(800,800)  


################# FUNCTIONS #################
directions='Use "+" and "-" keys to change Z in 10 um steps\nHold SHIFT and use "+" and "-" keys to change Z in 100 um steps\nPress "s" to save\nPress "q" to quit program'

def recoFrame(cropIM, z):
    #make even coordinates
    (yRez,xRez)=cropIM.shape
    if (xRez%2)==1:
        xRez-=1
    if (yRez%2)==1:
        yRez-=1
    cropIM=cropIM[0:yRez,0:xRez]
    complex = propagate(np.sqrt(cropIM), wvlen, z, dxy)	 #calculate wavefront at z
    amp = np.abs(complex)**2          # output is the complex field, still need to compute intensity via abs(res)**2
    ampInt = amp.astype('uint8')
    return(ampInt, complex)

def propagate(input_img, wvlen, zdist, dxy):
    M, N = input_img.shape # get image size, rows M, columns N, they must be even numbers!

    # prepare grid in frequency space with origin at 0,0
    _x1 = np.arange(0,N/2)
    _x2 = np.arange(N/2,0,-1)
    _y1 = np.arange(0,M/2)
    _y2 = np.arange(M/2,0,-1)
    _x  = np.concatenate([_x1, _x2])
    _y  = np.concatenate([_y1, _y2])
    x, y  = np.meshgrid(_x, _y)
    kx,ky = x / (dxy * N), y / (dxy * M)
    kxy2  = (kx * kx) + (ky * ky)

    # compute FT at z=0
    E0 = np.fft.fft2(np.fft.fftshift(input_img))

    # compute phase aberration
    _ph_abbr   = np.exp(-1j * np.pi * wvlen * zdist * kxy2)
    output_img = np.fft.ifftshift(np.fft.ifft2(E0 * _ph_abbr))
    return output_img

##############  MAIN  ##############

# find files in directory
print('Viewing image',imageFile)
z=defaultZ

# get image
im = cv2.imread(imageFile, 0) #Read the image as grayscale.              
count=0
done=False
print(directions)
while done==False:
    (ampIM, complexIM) = recoFrame(im, z*zScale)
    cv2.imshow('reco',cv2.resize(ampIM,DISPLAY_REZ))
    key=cv2.waitKey(1)
    if key==ord('='): 
        z+=zStep
        print(z,'um')
    elif key==ord('+'): 
        z+=zStep*10
        print(z,'um')
    elif key==ord('-'): 
        z-=zStep
        print(z,'um')
    elif key==ord('_'): 
        z-=zStep*10
        print(z,'um')
    elif key==ord('s'):
        saveName=imageFile[:-4]+'_'+str(z)+'.jpg'
        print('saving file',saveName)
        cv2.imwrite(saveName,ampIM)
        done=True
    elif key== ord('q'):
        done=True
        
cv2.destroyAllWindows()