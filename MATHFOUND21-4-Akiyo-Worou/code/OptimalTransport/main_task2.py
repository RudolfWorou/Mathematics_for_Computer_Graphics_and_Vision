import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
import cv2, os
def main():
    
   
    # Importing the images
    source_image=image_read_transform('src.png')
    target_image= image_read_transform('dst.png')
    print(target_image.shape)
    (m,n,col)=source_image.shape

    # resizing the images
    ratio=1    # change ratio to 4 for faster results
    source_image=cv2.resize(source_image,(int(n/ratio),int(m/ratio)))
    target_image=cv2.resize(target_image,(int(n/ratio),int(m/ratio)))

    (m,n,col)=source_image.shape 

   # cv2.imshow('source image',source_image)
    # cv2.imshow('target image',target_image)
    # cv2.waitKey(0)
    # creating the source and the target sets
    source_set= np.reshape(source_image,(m*n,col))
    target_set= np.reshape(target_image,(m*n,col))

    
    
    # source update
    new_source=np.reshape(source_update(source_set,target_set,5,100) , (m,n,col))
    # saving the image
    os.chdir(os.path.join(os.getcwd(),'OptimalTransport'))
    cv2.imwrite('result.jpg',255*new_source)
    cv2.imshow('result',new_source)
    cv2.waitKey(0)
    plt.show()
    
    




def image_read_transform(path):
    im=cv2.imread(os.path.join(os.getcwd(),'OptimalTransport',path),cv2.IMREAD_UNCHANGED)
    return im/255

def random_vector_3D():
    # we use spherical coordinates
    r=1  #radius or norm
    theta=2*np.pi*np.random.rand()  # longitude
    phi= np.pi*np.random.rand()  # colatitude 
    return np.array([r*np.cos(theta)*np.sin(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(phi)])
    
def projection(set1,set2,theta): 
    # We assume that set1 and set2 have the same length
    L=[];G=[]
    for i in range(len(set1)):
        L.append(np.inner( set1[i],theta))  
        G.append(np.inner( set2[i],theta))
    return np.array(L), np.array(G)
 
def source_update(set1,set2,directions=5,iterations=100):  
    
    new_source=set1.copy()
    for _ in range(iterations):
        
        n_s=np.zeros_like(set1,dtype=float)
        for _ in range(directions):
            # random theta
            theta=random_vector_3D()
            # projection
            proj_set1,proj_set2 = projection(new_source,set2,theta)
            # sort
            source_id=np.argsort(proj_set1)
            target_id = np.argsort(proj_set2)
            # new source
            for j in range(len(target_id)):
                n_s[source_id[j]] += (proj_set2[target_id[j]]-proj_set1[source_id[j]])*theta
    
            
        new_source+=n_s/directions
    

    return new_source 













    

if __name__=='__main__':
    main()