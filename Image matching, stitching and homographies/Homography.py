import numpy as np
import skimage.color
import skimage.io
from scipy import linalg
import skimage.transform
from q2 import briefLite,briefMatch,plotMatches
import matplotlib.pyplot as plt



# Q 3.1
def computeH(l1, l2):
   aList=[]
   for i in range(l1.shape[0]):
        x=l1[i,0]
        y=l1[i,1]
        u=l2[i,0]
        v=l2[i,1]
        a1=[0, 0, 0,-u,-v,-1,y*u,y*v,y]
        a2=[u,v,1,0,0,0,-x*u,-x*v,-x]
        aList.append(a1)
        aList.append(a2)
   A=np.matrix(aList)
   (U,S,V) = np.linalg.svd(A)
   V=np.transpose(V)
   H2to1 = np.reshape(V[:,8],(3,3))
   H2to1=H2to1/H2to1[2,2]
   return H2to1

# Q 3.2
def computeHnorm(x1, x2):#N*2
    H2to1 = np.eye(3)
    maxstd = np.sqrt(2) # the largest distance to the origin is √2
    x1_ones = np.ones((np.shape(x1)[0],3))
    x2_ones = np.ones((np.shape(x2)[0],3))
    x1_ones[:,0:2] = x1
    x2_ones[:,0:2] = x2
    x1 = x1_ones.transpose()
    x2 = x2_ones.transpose()
    # condition points (important for numerical reasons)
    # --from points--
    m = np.mean(x2[:2], axis=1)
    # maxstd = max(std(x2[:2], axis=1)) + 1e-9
    C2 = np.diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    x2 = np.dot(C2,x2)
    # --to points--
    m = np.mean(x1[:2], axis=1)
    #maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C1 = np.diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    x1 = np.dot(C1,x1)
    H2to1 = computeH(x1[:2].transpose(), x2[:2].transpose())
    # decondition
    H2to1 = np.dot(linalg.inv(C1),np.dot(H2to1,C2))
    # normalize and return
    return H2to1 /H2to1[2,2]

# Q 3.3
def computeHransac(locs1, locs2):
    bestH2to1, inliers = None, None
    # YOUR CODE HERE
   	#swap x column and y column
    #find 4 random points to calculate H
        #find 4 random points to calculate H
    pointset1=locs1
    pointset2=locs2#N*2
    #create a set of random index to match points
    randIdx=np.random.randint(0,pointset1.shape[0]-1,size=5000*4,dtype=int)
    randIdx=randIdx.reshape(5000,4)
    TransPoint=np.zeros([2,locs1.shape[0]])
    num=[]
    H_list=[]
    for i in range(5000):
        index=randIdx[i,:]
        p1=pointset1[index,:]
        p2=pointset2[index,:]#4*2
        H=computeHnorm(p1, p2)
        H_list.append(H)
        num_lniers=0
        inliers_each=[]
        for m in range(pointset1.shape[0]):
            p=np.transpose(np.matrix([pointset2[m,0],pointset2[m,1],1]))
            trans_point=np.dot(H,p)
            trans_point=trans_point/trans_point[2,0]
            TransPoint=np.transpose([trans_point[0,0],trans_point[1,0]])
            dist = np.linalg.norm(pointset1[m,:]-TransPoint)
            if dist<2:
                num_lniers=num_lniers+1
                inliers_each.append(1)
            else:
                inliers_each.append(0)
        num.append(num_lniers)
        if num[i]>num[i-1]:
            inliers=inliers_each
        #print(i,':',num_lniers)
    num=np.asarray(num)
    maxnum_index=np.argmax(num)
    bestH=H_list[maxnum_index]
    print('max num of inlier points',max(num))
    print('best H:\n',bestH)
    return bestH,inliers

# Q3.4
# skimage.transform.warp will help
def compositeH( H2to1, template, img ):
    # YOUR CODE HERE
    #template_image = Image.open(template)
    #a_band = template_image.split()[-1]
    #template_size = template_image.size
    M=np.linalg.inv(H2to1)
    M=M/M[2,2]
    wrap_img=skimage.transform.warp(img,H2to1,output_shape=template.shape)
    #plt.imshow(wrap_img)
    compositeimg=template
    i1x,i1y=wrap_img.shape[:2]
    for i in range(i1x):
        for j in range(i1y):
            A=wrap_img[i,j]
            if not (np.array_equal(A,np.array([0,0,0]))):
                    #bl,gl,rl = wrap_img[i,j]
                    compositeimg[i,j]=255*wrap_img[i,j]
    #compositeimg=skimage.transform.warp(img,H)
    return compositeimg


def HarryPotterize():
    # we use ORB descriptors but you can use something else
    #from skimage.feature import ORB,match_descriptors,plot_matches
    # YOUR CODE HERE
    img=skimage.io.imread('../data/cv_cover.jpg')
    img1=skimage.color.rgb2gray(img)
    imgreference=skimage.io.imread('../data/cv_desk.png')
    #skimage.io.imshow(imgreference)
    imgreference=skimage.io.imsave('../data/cv_desk.jpg', imgreference)
    imgreference=skimage.io.imread('../data/cv_desk.jpg')
    img2=skimage.color.rgb2gray(imgreference)
    imgread=skimage.io.imread('../data/hp_cover.jpg')
    #imgread=skimage.color.rgb2gray(imgread)
    imgread=skimage.transform.resize(imgread,img1.shape)
    locs1,desc1=briefLite(img1)
    locs2,desc2=briefLite(img2)
    match=briefMatch(desc1,desc2,ratio=0.8)
    x1=locs1[match[:,0],:]
    x2=locs2[match[:,1],:]
    x1[:,[0,1]] = x1[:,[1,0]]
    x2[:,[0,1]] = x2[:,[1,0]]
    H,inliners=computeHransac(x1,x2)
    #wrap_img=skimage.transform.warp(imgread,H,output_shape=img2.shape)
    img_result=compositeH(H,imgreference,imgread)
    #imgresult=skimage.color.gray2rgb(img_result)
    plt.imshow(img_result)
    plt.show()
    return
