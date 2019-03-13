import numpy as np
from scipy.optimize import fsolve

# Q 2.1
def eightpoint(pts1, pts2, M=1):
    F = None
    #scale=np.matrix([[1/M,0,0],[0,1/M,0],[0,0,1]])
    n = pts1.shape[0]
    x1=np.matrix(np.ones([3,n]))
    x1[0,:]=pts1[:,0]
    x1[1,:]=pts1[:,1]
    x2=np.matrix(np.ones([3,n]))
    x2[0,:]=pts2[:,0]
    x2[1,:]=pts2[:,1]
    mean_1 = np.mean(x1[:2],axis=1)
    S1 =2/M #/ np.std(x1[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    #T1 = np.array([[S1,0,-1],[0,S1,-1],[0,0,1]])
    x1 = np.dot(T1,x1)

    #x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = 2/M #/ np.std(x2[:2])
    #T2 = np.array([[S2,0,-1],[0,S2,-1],[0,0,1]])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2 = np.dot(T2,x2)
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
        '''A[i] = [pts1[i,0]*pts2[i,0], pts1[i,0]*pts2[i,1], pts1[i,0],
                pts1[i,1]*pts2[i,0], pts1[i,1]*pts2[i,1], pts1[i,1],
                pts2[i,0], pts2[i,1],1]'''

    # compute linear least square solution
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U,np.dot(np.diag(S),V))
    F=np.dot(T1.T,np.dot(F,T2))
    F=np.transpose(F)
    F=F/F[2,2]
    return F


# Q 2.2
# you'll probably want fsolve
def sevenpoint(pts1, pts2, M=1):
    F = None
    n = pts1.shape[0]
    x1=np.matrix(np.ones([3,n]))
    x1[0,:]=pts1[:,0]
    x1[1,:]=pts1[:,1]
    x2=np.matrix(np.ones([3,n]))
    x2[0,:]=pts2[:,0]
    x2[1,:]=pts2[:,1]
    mean_1 = np.mean(x1[:2],axis=1)
    #S1 =1/np.sqrt(M) #/ np.std(x1[:2])
    S1=2/M
    T1 = np.array([[S1,0,-1],[0,S1,-1],[0,0,1]])
    x1 = np.dot(T1,x1)

    #x2 = x2 / x2[2]
    mean_2 = np.mean(x2[:2],axis=1)
    S2 = 2/M #/ np.std(x2[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    #T2=np.array([[S2,0,-1],[0,S2,-1],[0,0,1]])
    x2 = np.dot(T2,x2)
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
    # compute linear least square solution
    U,S,V = np.linalg.svd(A)
    F8 = V[7].reshape(3,3)
    F9= V[8].reshape(3,3)
    def funC(alpha):
        return np.linalg.det(alpha*F8 + (1-alpha)*F9)
    f=np.real(fsolve(funC,0.001))
    for iCell in range(len(f)):
        F = f[iCell]*F8 + (1-f[iCell])*F9;
        F=np.dot(T1.T,np.dot(F,T2))
        F=np.transpose(F)
    return F
