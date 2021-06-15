import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from tkinter import *
from PIL import Image
from scipy.ndimage import gaussian_filter
from graph_cut import GraphCut
from graph_cut_gui import GraphCutGui
import math
import os

class GraphCutController:

    def __init__(self):
        self.__init_view()

    def __init_view(self):
        root = Tk()
        root.geometry("700x500")
        self._view = GraphCutGui(self, root)
        root.mainloop()

    # TODO: TASK 2.1
    def __get_color_histogram(self, image, seed, hist_res):
        """
	Compute a color histograms based on selected points from an image
	
	:param image: color image
	:param seed: Nx2 matrix containing the the position of pixels which will be
	            used to compute the color histogram
	:param histRes: resolution of the histogram
	:return hist: color histogram
        """
        
        L=[]
        N=len(seed)
        for i in range(N):
            
            L.append(image[seed[i][1],seed[i][0]])
        image_part=np.array(L)
        
        
        hist, bins= np.histogramdd(image_part,bins=hist_res,range=((0,255),(0,255),(0,255)) )
        #hist= ndimage.gaussian_filter(hist,sigma=7)   # Gaussian smoothing

        return hist /np.linalg.norm(hist)


    # TODO: TASK 2.2
    # Hint: Set K very high using numpy's inf parameter
    def __get_unaries(self, image, lambda_param, hist_fg, hist_bg, seed_fg, seed_bg):
        """

        :param image: color image as a numpy array
        :param lambda_param: lamdba as set by the user
        :param hist_fg: foreground color histogram
        :param hist_bg: background color histogram
        :param seed_fg: pixels marked as foreground by the user
        :param seed_bg: pixels marked as background by the user
        :return: unaries : Nx2 numpy array containing the unary cost for every pixels in I (N = number of pixels in I)
        """
    
        hist_res= hist_fg.shape[0]
        K= np.inf
        (n,m,_) = image.shape
        L=[]
        for i in range(n):
            for j in range(m):

                pixel_pos=[j,i]
                if pixel_pos in seed_bg :
                    L.append( [0,K])   # set unaries 
                elif pixel_pos in seed_fg:
                    L.append([K,0])   
                else:

                    pixel_cord_in_hist= list(map(int, (hist_res/256)*np.array(  image[i,j] ) )) # pixel position in the histogram
                   
                    L.append( [-lambda_param*np.log(hist_fg[ tuple(pixel_cord_in_hist) ] +10e-10 ),-lambda_param*np.log(hist_bg[tuple(pixel_cord_in_hist)]+10e-10) ]  )
                
        return np.array(L)




        

    # TODO: TASK 2.3
    # Hint: Use coo_matrix from the scipy.sparse library to initialize large matrices
    # The coo_matrix has the following syntax for initialization: coo_matrix((data, (row, col)), shape=(width, height))
    def __get_pairwise(self, image):
        """
        Get pairwise terms for each pairs of pixels on image
        :param image: color image as a numpy array
        :return: pairwise : sparse square matrix containing the pairwise costs for image
        """
        # We will use the adjacency matrix

        P=[]
        (n,m, _)= image.shape

        sigma=5
        def B(p,q):
            Ip=np.mean(p) # Itensity
            Iq=np.mean(q)
             
            return np.exp(-0.5*((Ip-Iq)/sigma)**2 ) 
            
        L=[]
        for i in range(n):
            for j in range(m):
                L.append(image[i,j])
        image_flat=np.array(L)
        for i in range(n*m-m):
            B01=B(image_flat[i], image_flat[i+1])
            B01_=B(image_flat[i], image_flat[i+m])
            
            P.append([i,i+m,0,B01_,B01_,0]) # vertice with the node below
            if i %(m-1)!=0:   # Check if it is not the end of a line
                P.append([i,i+1,0,B01,B01,0])   # vertice with the node on the right 
             
        for i in range(n*m-m,n*m-1):
            B01__=B(image_flat[i], image_flat[i+1])
            P.append([i,i+1,0,B01__,B01__,0])

       
        return np.array(P)


        
    # TODO TASK 2.4 get segmented image to the view
    def __get_segmented_image(self, image, labels, background=None):
        """
        Return a segmented image, as well as an image with new background 
        :param image: color image as a numpy array
        :param label: labels a numpy array
        :param background: color image as a numpy array
        :return image_segmented: image as a numpy array with red foreground, blue background
        :return image_with_background: image as a numpy array with changed background if any (None if not)
        """
        (n,m,_)=image.shape
        L=[]
        for i in range(n):
            for j in range(m):
                L.append(image[i,j])
        image_flat=np.array(L)
        image_segmented=np.zeros_like(image_flat)
        for i in range(len(labels)):
            px=image_flat[i]
            if labels[i]==0:
                image_segmented[i]=[px[0],0,0]
            elif labels[i]==1:
                image_segmented[i]=[0,0,205]
        image_segmented=np.reshape(image_segmented,(n,m,3))

        if background is None:
            return image_segmented,None
        else:
            (p,q,_)=background.shape
            G=[]
            for i in range(p):
                for j in range(q):
                    G.append(background[i,j])
            background_flat=np.array(G)

            image_with_background=np.zeros_like(image_flat)
            for i in range(len(labels)):
                px=image_flat[i]
                if labels[i]==0:
                    image_with_background[i]=px
                elif labels[i]==1:
                    image_with_background[i]=background_flat[i]
            
            image_with_background=np.reshape(image_with_background,(p,q,3))

            return image_segmented,image_with_background






    def segment_image(self, image, seed_fg, seed_bg, lambda_value, background=None):
        image_array = np.asarray(image)
        background_array = None
        if background:
            background_array = np.asarray(background)
        
        height, width = np.shape(image_array)[0:2]
        num_pixels = height * width

        # TODO: TASK 2.1 - get the color histogram for the unaries
        hist_res = 32
        cost_fg = self.__get_color_histogram(image_array, seed_fg, hist_res)
        cost_bg = self.__get_color_histogram(image_array, seed_bg, hist_res)

        # TODO: TASK 2.2-2.3 - set the unaries and the pairwise terms
        unaries = self.__get_unaries(image_array, lambda_value, cost_fg, cost_bg, seed_fg, seed_bg)
        pairwise = self.__get_pairwise(image_array)

        # TODO: TASK 2.4 - perform graph cut
        # Your code here
        connections= 4*num_pixels
        nodes = num_pixels
        graph=GraphCut(num_pixels,connections)
        graph.set_unary(unaries)
        graph.set_pairwise(pairwise)
        graph.minimize()
        labels=graph.get_labeling()


        # TODO TASK 2.4 get segmented image to the view
        segmented_image, segmented_image_with_background = self.__get_segmented_image(image_array, labels,
                                                                                      background_array)
        # transform image array to an rgb image
        segmented_image = Image.fromarray(segmented_image, 'RGB')
        # saving the image
        segmented_image.save(os.path.join(os.getcwd(),'GraphCuts','result_images','segmented_image.jpg'))

        self._view.set_canvas_image(segmented_image)
        if segmented_image_with_background is not None:
            segmented_image_with_background = Image.fromarray(segmented_image_with_background, 'RGB')
            # saving the image
            segmented_image_with_background.save(os.path.join(os.getcwd(),'GraphCuts','result_images','segmented_image_with_background.jpg'))
            
            plt.imshow(segmented_image_with_background)
            plt.show()
