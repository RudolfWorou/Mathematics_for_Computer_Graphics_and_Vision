import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats

def main():
    
    n=10000
    mu_1,sigma_1=25,5
    mu_2,sigma_2 = 100, 10
    N_1,bins_1,kde_pdf_1=dist_bins_kde(mu_1, sigma_1,n)
    N_2,bins_2,kde_pdf_2=dist_bins_kde(mu_2, sigma_2,n)
    
    # Optimal transport computation
    OT=np.sum(np.array(sorted(N_2))-np.array(sorted(N_1)) )/n
    print("The optimal transport is ",OT)

    # Closed form solution for Optimal transport
    W_22=mu_2-mu_1+sigma_1+sigma_2-2*np.sqrt(sigma_1*sigma_2)

    print("The Closed value to the optimal transport is : ",W_22)

    # Error 
    print(" The error of approximation is: ",abs((W_22-OT)/OT))
    # Interpolated distribution
    L=[0.25,0.5,0.75]
    dist_list=[]
    for l in L:
        N=l*N_1+(1-l)*N_2
        pdf=stats.gaussian_kde(N)              # Using Gaussian kernel
        mu=l*mu_1+(1-l)*mu_2
        sigma= np.sqrt( (l*sigma_1)**2+((1-l)*sigma_2 )**2)
        bins=np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
        kde_pdf=pdf.evaluate(bins)   
        dist_list.append( (N,bins,kde_pdf) )
        
    



    # Distribution plot
    for element in dist_list:
        plt.scatter(element[1],element[2],linewidths=0.02,c='orange')
    
    plt.scatter(bins_1,kde_pdf_1,linewidths=0.02,c='red')
    plt.scatter(bins_2,kde_pdf_2,linewidths=0.02,c='blue')
    plt.show()

def dist_bins_kde(mu, sigma,n) : 
    np.random.seed(99)
    N=np.random.normal(mu,sigma,n)
    pdf=stats.gaussian_kde(N)              # Using Gaussian kernel
    bins=np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
    kde_pdf=pdf.evaluate(bins)                 
    return N, bins, kde_pdf

if __name__ == '__main__':
    main()
