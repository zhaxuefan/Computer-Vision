import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def camera2(E):
    U,S,Vt = linalg.svd(E)
    m=(S[0]+S[1])/2.0
    E = U.dot(np.diag([m,m,0])).dot(Vt)
    U,S,Vt = linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    if np.linalg.det(U.dot(W).dot(Vt)) < 0:
        W = -W
    M2s = []
    t = (U[:,2]/abs(U[:,2]).max()).reshape((3,1))
    M2s.append(np.hstack([U.dot(W).dot(Vt),t]))
    M2s.append(np.hstack([U.dot(W).dot(Vt),-t]))
    M2s.append(np.hstack([U.dot(W.T).dot(Vt),t]))
    M2s.append(np.hstack([U.dot(W.T).dot(Vt),-t]))

    M2s = np.array(M2s)
    return M2s


def plot_epipolar_line(im,F,pt):
    """ Plot the epipole and epipolar line F*x=0
        in an image. F is the fundamental matrix 
        and x a point in the other image."""
    
    m,n = im.shape[:2]
    x = np.array([pt[0],pt[1],1])
    line = np.dot(F,x)
    
    # epipolar line parameter and values
    t = np.linspace(0,n,100)
    lt = np.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])

    # take only line points inside the image
    #ndx = (lt>=0) & (lt<m) 
    # we'll use the whole line
    plt.plot(t,lt,linewidth=2)
    
    # don't need this!
    #U,S,V = linalg.svd(F)
    #e = V[-1]
    #epipole = e/e[2]
    #plt.plot(epipole[0]/epipole[2],epipole[1]/epipole[2],'r*')

def plot_epipolar_lines(im1,im2,F,pts1,idxs_to_plot=[1,5,9]):
    plt.subplot(121)
    plt.imshow(im1)
    for idx in idxs_to_plot:
        plt.plot(pts1[idx,0],pts1[idx,1],ms=10,marker='o')
    plt.subplot(122)
    for idx in idxs_to_plot:
        plot_epipolar_line(im2,F,pts1[idx,:])
    plt.imshow(im2)
    plt.show()

    
def plot_matched_points(im1,im2,F,pts1,pts2,pts2e):
    plt.subplot(221)
    plt.imshow(im1)
    plt.title('source')
    for pt in pts1:
        plt.plot(pt[0],pt[1],ms=5,marker='o')

    plt.subplot(222)
    plt.imshow(im2)
    plt.title('given')
    for pt in pts2:
        plt.plot(pt[0],pt[1],ms=5,marker='o')

    plt.subplot(223)
    for pt in pts1:
        plot_epipolar_line(im2,F,pt)
    plt.imshow(im2)
    plt.title('lines')

    plt.subplot(224)
    plt.imshow(im2)
    plt.title('matched')
    for pt in pts2e:
        plt.plot(pt[0],pt[1],ms=5,marker='o')
    plt.show()