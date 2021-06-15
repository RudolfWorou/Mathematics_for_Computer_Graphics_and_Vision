import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
def local_laplacian(im,x,y):
    n,m = im.shape
    # Neumann condition : adding a 1
    try:
        top = im[x-1,y]
    except:
        top = 1
    try:
        bottom = im[x+1,y]
    except:
        bottom = 1
    try:
        right = im[x,y+1]
    except:
        right = 1
    try:
        left = im[x,y-1]
    except:
        left = 1
    
            
    return top+bottom+right+left-4*im[x,y]
    

def Laplacian(im):         # Global transformation
    out_im = np.zeros_like(im)
    n,m = im.shape
    for x in range(n):
        for y in range(m):
            out_im[x,y]=local_laplacian(im,x,y)
    return out_im

def heat_diffusion(im,original_im,time_step,iterations=[25,50,75,100]):
    m=max(iterations)
    final_im=im.copy()
    error=[np.sum(np.abs(original_im-final_im))]  # initial error  
    for i in range(1,m+1):
        final_im+=time_step*Laplacian(final_im)

        error.append(np.sum(np.abs(original_im-final_im)))

        if i in iterations:
            im=Image.fromarray((np.clip(final_im*255.0, 0, 255)).astype(np.uint8))
            # saving the image
            im_name='diffusion_' +str(i) + '.jpg'
            im.save( os.path.join(os.getcwd(),'code','result_images',im_name),'JPEG' ) 

            im.show()
    
    error=(1/(np.max(error)-np.min(error)))*(error-np.min(error))  # error normalization
    #print(error)
    # curve smoothing
    plt.xticks(np.linspace(0,m,int(m/10)+1) )
    plt.plot(error )
    plt.xlabel('Evolution of the error over iteration')
    plt.ylabel('Errors')
    plt.title('Evolution of the errors over the iterations')
    plt.show()