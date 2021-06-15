import numpy as np 
from PIL import Image
import math
from scipy.ndimage import convolve
from task1 import *
from task2 import *
from task3 import*
import os
def main():

    I_orig_img = Image.open('lotr.jpg')
    I_orig_img = I_orig_img.convert('L')
    I_orig = np.array(I_orig_img)/255.0

    h, w = np.shape(I_orig)
    
    # Add noise
    gauss = np.random.normal(0, np.sqrt(0.05),(h, w))  # the variance is 0.05, a sqare root was forgotten
    gauss = gauss.reshape(h, w)
    I_n = I_orig + gauss


    
    # Visualize
    I_orig_img.show()
    Image.fromarray((np.clip(I_n*255.0, 0, 255)).astype(np.uint8)).show()

    # used to save the noised image
        #Image.fromarray((np.clip(I_n*255.0, 0, 255)).astype(np.uint8)).save( os.path.join(os.getcwd(),'code','result_images','noisy_image.jpg'),'JPEG' )

    # Task 1 : Image filtering

    image_filtering(I_n,I_orig)

    # Task 2 : Heat diffusion

    heat_diffusion(I_n,I_orig,0.08)  # time step chosen to match the given figure 

    # Task 3 : Variational method
    a=9
    b=-2
    variational(I_n,a,b)


if __name__ == "__main__":
    main()


