from graph_cut import GraphCut
import numpy as np
import scipy as sp

def main():
    
    # Graph creation
    graph=GraphCut(3,4)
    #graph.set_neighbors(sp.sparse.csr_matrix([[0,3,0],[2,0,5],[0,1,0]]))

    # Using unaries
    U=np.array([[4,9],[7,7],[8,5]])
    graph.set_unary(U)
    # Pairwise  # [i,j,e00,e01,e10,e11]
    P=[[0,1,0,3,2,0],[1,2,0,5,1,0] ]
    graph.set_pairwise(P)

    # Minimization result
    print(graph.minimize() )
    # labels for the graph
    print(graph.get_labeling())
    


if __name__ == '__main__':
    main()
