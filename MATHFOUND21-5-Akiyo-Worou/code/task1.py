import math
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
def apply_gaussian_filter(im,kernel):
    kernel_size=len(kernel)
    
    # kernel center is in 0,0
    
    # creating an image with more value at the edges 
    
    left_im=np.ones_like(im[:,0])
    right_im=np.ones_like(im[:,-1])
    im_1 = im.copy() 
    for i in range(kernel_size):
        im_1 = np.c_[left_im,im_1,right_im]   # Adding more elements on the left and the right of  the image
    gauss_im = im_1.copy()
    top_im= np.ones_like(gauss_im[0,:])
    bottom_im = np.ones_like(gauss_im[-1,:])

    for i in range(kernel_size):
        gauss_im = np.r_[ [top_im] ,gauss_im, [bottom_im] ]  # adding elements on the top and on the bottom
    
    # By using the Neumann condition we maintain the boundaries pixel values
    n, m = np.shape(gauss_im)

    
    filtered_gauss_im=gauss_im.copy()
    for i in range(kernel_size,n-kernel_size):
        for j in range(kernel_size,m-kernel_size):
            # get the kernel mapping array
            im_part= gauss_im[:,j:j+kernel_size]
            im_sub= im_part[i:i+ kernel_size,:]

            filtered_gauss_im[i,j]= np.sum( np.multiply( im_sub,kernel)) # applying the convolution
    return filtered_gauss_im[kernel_size:n-kernel_size,kernel_size:m-kernel_size]

def gaussian_filter(sigma):
    kernel_size=2*math.ceil(3*sigma)+1
    kernel=[[0 for i in range(kernel_size) ] for j in range(kernel_size) ] # kernel initialization
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[x][y]=np.exp(- (x**2+y**2)/(2*(sigma)**2) )  # Gaussian kernel expression without normalization
    kernel= np.array(kernel)
    kernel=(1/np.sum(kernel))*kernel
    return kernel

def image_filtering(im,original_im,sigma=0.5,iter_stop=[8,16,24,32]):
    
    kernel=gaussian_filter(sigma)
    m=max(iter_stop)
    final_im=im.copy()
    error=[np.sum(np.abs(original_im-final_im))]  # initial error
    for i in range(1,m+1):
        final_im=apply_gaussian_filter(final_im, kernel) 
        error.append(np.sum(np.abs(original_im-final_im)))
        if i in iter_stop:
            im=Image.fromarray((np.clip(final_im*255.0, 0, 255)).astype(np.uint8))
            # saving the image
            im_name='filtered_' +str(i) + '.jpg'
            im.save( os.path.join(os.getcwd(),'code','result_images',im_name),'JPEG' ) 

            im.show()
    
    error=(1/(np.max(error)-np.min(error)))*(error-np.min(error))  # error normalization
    #print(error)
    plt.plot(error)
    plt.xlabel('Number of filtering')
    plt.ylabel('Errors')
    plt.title('Evolution of the errors over the number of filtering')
    plt.show()
