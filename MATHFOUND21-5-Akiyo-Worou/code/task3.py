import numpy as np
import os
from PIL import Image
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
def variational(im,a,b):
    H,W= im.shape

    # image vectorization
    Ivec=np.reshape(im.T,(H*W,1))

    # creating the matrix A_lambda,B_lambda and C_lambda
    B_lambda =lil_matrix((H,H))
    for i in range(H-1):
        B_lambda[i,i+1]=b

    C_lambda=lil_matrix((H,H))
    for i in range(1,H):
        C_lambda[i,i-1]=b

    A_lambda=np.diag([a for i in range(H)]) + B_lambda+C_lambda

    # Creating A
    A= lil_matrix( (H*W,H*W))

    
    A[(W-1)*H:W*H,(W-1)*H:W*H]=A_lambda

    for i in range(0,W-1):
        A[(i+1)*H:(i+2)*H,i*H:(i+1)*H]=B_lambda
        A[i*H:(i+1)*H,(i+1)*H:(i+2)*H]=C_lambda
        A[i*H:(i+1)*H,i*H:(i+1)*H]=A_lambda

    # solving the equation
    
    I_uvec=spsolve(A,Ivec)
    
    # reshape to get the image
    final_im=np.reshape(I_uvec,(W,H)).T

    # 
    out_im=Image.fromarray((np.clip(final_im*255.0, 0, 255)).astype(np.uint8))
    # saving the image
    im_name='variational.jpg'
    out_im.save( os.path.join(os.getcwd(),'code','result_images',im_name),'JPEG' ) 

    out_im.show()
