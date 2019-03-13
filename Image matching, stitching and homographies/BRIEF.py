import numpy as np
import scipy.io as sio
import skimage.feature
import matplotlib.pyplot as plt
import skimage.transform
# Q2.1
# create a 2 x nbits sampling of integers from to to patchWidth^2
# read BRIEF paper for different sampling methods
def makeTestPattern(patchWidth, nbits):
    res = None
    # YOUR CODE HERE
    compareX=[]
    compareY=[]
    for i in range(nbits):
        x=np.random.randint(0,patchWidth**2-1)
        y=np.random.randint(0,patchWidth**2-1)
        compareX.append(x)
        compareY.append(y)
    compareX=np.reshape(compareX,(nbits,1))
    compareY=np.reshape(compareY,(nbits,1))
    res=[compareX,compareY]
    return res 

# Q2.2
# im is 1 channel image, locs are locations
# compareX and compareY are idx in patchWidth^2
# should return the new set of locs and their descs
def computeBrief(im,locs,compareX,compareY,patchWidth=9):
    desc = None
    # YOUR CODE HERE
    locsnew=[]
    for m in range(len(locs)):
        if(locs[m,0]>=4 and locs[m,0]<im.shape[0]-4 and locs[m,1]>=4 and locs[m,1]<im.shape[1]-4):
            locsnew.append(locs[m])
    locsnew=np.asarray(locsnew)
    desc =np.zeros([len(locsnew),len(compareX)])
    for m in range(len(locsnew)):
        for n in range(len(compareX)):
            x1=compareX[n]//9
            y1=compareX[n]-x1*9
            x2=compareY[n]//9
            y2=compareY[n]-x2*9
            x1=x1-4
            x2=x2-4
            y1=y1-4
            y2=y2-4
            desc[m,n]=im[int(locsnew[m,0]+x1),int(locsnew[m,1]+y1)]<im[int(locsnew[m,0]+x2),int(locsnew[m,1]+y2)]
    locs=locsnew
    return locs, desc

# Q2.3
# im is a 1 channel image
# locs are locations
# descs are descriptors
# if using Harris corners, use a sigma of 1.5
def briefLite(im):
    locs, desc = None, None
    # YOUR CODE HERE
    x=sio.loadmat('testPattern.mat')
    compareX=x['compareX']
    compareY=x['compareY']
    locs = skimage.feature.corner_peaks(skimage.feature.corner_harris(im,1.5), min_distance=1,num_peaks=1000)
    locs, desc = computeBrief(im,locs,compareX,compareY)
    return locs, desc

# Q 2.4
def briefMatch(desc1,desc2,ratio=0.8):
    # okay so we say we SAY we use the ratio test
    # which SIFT does
    # but come on, I (your humble TA), don't want to.
    # ensuring bijection is almost as good
    # maybe better
    # trust me
    matches = skimage.feature.match_descriptors(desc1,desc2,'hamming',cross_check=True)
    return matches

def plotMatches(im1,im2,matches,locs1,locs2):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    skimage.feature.plot_matches(ax,im1,im2,locs1,locs2,matches,matches_color='r')
    plt.show()
    return

def testMatch(im1,im2):
    # YOUR CODE HERE
    locs1,desc1=briefLite(im1)
    locs2,desc2=briefLite(im2)
    matches=briefMatch(desc1,desc2,ratio=0.8)
    #print(len(matches))
    plotMatches(im1,im2,matches,locs1,locs2)
    return locs1,locs2,matches


# Q 2.5
# we're also going to use this to test our
# extra-credit rotational code
#def briefRotTest(briefFunc=briefLite):
def briefRotTest():
    # you'll want this
    # YOUR CODE HERE
    img = skimage.io.imread('../data/model_chickenbroth.jpg')
    im = skimage.color.rgb2gray(img)
    m=[]
    num=[]
    xcenter=im.shape[0]/2
    ycenter=im.shape[1]/2
    for i in range(0,360,10):
        m.append(i)
        imnew=skimage.transform.rotate(im,i)
        locs1,locs2,matches=testMatch(imnew,im)
        imnew_match=locs1[matches[:,0],:]
        im_match=locs2[matches[:,1],:]
        im_estimate_x=(im_match[:,0]-xcenter)*np.cos(i*np.pi/180)-(im_match[:,1]-ycenter)*np.sin(i*np.pi/180)+xcenter
        im_estimate_y=(im_match[:,0]-xcenter)*np.sin(i*np.pi/180)+(im_match[:,1]-ycenter)*np.cos(i*np.pi/180)+ycenter
        a=0
        for j in range(len(imnew_match)):
            if round(im_estimate_x[j])==imnew_match[j,0] and round(im_estimate_y[j])==imnew_match[j,1]:
                a=a+1
        num.append(a)
    plt.bar(m,num)
    plt.show()
    return

# Q2.6
# YOUR CODE HERE


# put your rotationally invariant briefLite() function here
def briefRotLite(im):
    locs, desc = None, None
    # YOUR CODE HERE
    return locs, desc