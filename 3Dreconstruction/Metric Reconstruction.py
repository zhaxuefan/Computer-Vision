import numpy as np
import q2

# Q3.1
def essentialMatrix(F,K1,K2):
    E = None
    E=np.dot(np.transpose(K2),F)
    E=np.dot(E,K1)
    return E

# Q3.2

def triangulate(P1, pts1, P2, pts2):
    P, err = None, None
    num=pts1.shape[0]
    P=np.zeros((num,3))
    err=np.zeros(num)
    A=np.zeros((4,4))

    for k in range(num):
        x,y=pts1[k,:]
        xp,yp=pts2[k,:]

        A[0,:]=y*P1[2,:]-    P1[1,:]
        A[1,:]=  P1[0,:]-  x*P1[2,:]
        A[2,:]=yp*P2[2,:] -  P2[1,:]
        A[3,:]=   P2[0,:]-xp*P2[2,:]

        U, S, V = np.linalg.svd(A)

        point=V[-1,:]
        point=point/point[-1]


        P[k,:]=point[0:-1]

        mm1=P1.dot(point)
        mm1=mm1/mm1[-1]

        mm2=P2.dot(point)
        mm2=mm2/mm2[-1]

        err[k]=np.linalg.norm(pts1[k,:]-mm1[0:-1])+ np.linalg.norm(pts2[k,:]-mm2[0:-1])

    err=np.sum(err)
    return P, err