import numpy as np
from q2 import eightpoint, sevenpoint
from q3 import triangulate

import scipy.optimize
# Q 5.1
# we're going to also return inliers
def ransacF(pts1, pts2, M):
    #根据RANSAC随机找点然后计算
    num=pts1.shape[0] #个数
    
    selnum=10 #计算F所需的最小点数
#    Error=np.zeros(num)   #计算误差
    
    thed=0.1 #判断再里面的阈值
    maxgen=1000 #最大500次
    
    bestinlinernum=0
    bestF=0
    inliers=0
    
    for k in range(maxgen): #每一次寻找
        idxs=np.random.choice(num,selnum,replace=False) #不放回采样
        selpts1=pts1[idxs,:]
        selpts2=pts2[idxs,:]

        selF=eightpoint(selpts1, selpts2, M) #计算F
        selF = selF/selF[2,2]
        
        curinliers=mycalc(pts1,pts2,selF)  # 所有点
        
        flag= abs(curinliers)<thed
        
        curinlinernum=np.sum(flag+0) #计算当前内点个数
        
        if curinlinernum > bestinlinernum:
            bestF=selF
            inliers=flag
            bestinlinernum=curinlinernum

    
    return bestF, inliers

# Q 5.2
# r is a unit vector scaled by a rotation around that axis
# 3x3 rotatation matrix is R
# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# http://www.cs.rpi.edu/~trink/Courses/RobotManipulation/lectures/lecture6.pdf
def rodrigues(r): #给定向量生成旋转矩阵
    theta=np.linalg.norm(r)
    k=r/theta
    kx,ky,kz=k
    K=np.array([[0,-kz,ky],[kz,0,-kx],[-ky,kx,0]])
    I=np.eye(3)
    R=I+np.sin(theta)*K+(1-np.cos(theta))*(K@K)
    
    return R


# Q 5.2  binggo
# rotations about x,y,z is r
# 3x3 rotatation matrix is R
# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
#http://www.cs.rpi.edu/~trink/Courses/RobotManipulation/lectures/lecture7.pdf
def invRodrigues(R):
    r11,r12,r13=R[0,:]
    r21,r22,r23=R[1,:]
    r31,r32,r33=R[2,:]

    q0=np.sqrt((1+r11+r22+r33)/4)
    q1=np.sqrt((1+r11-r22-r33)/4)
    q2=np.sqrt((1-r11+r22-r33)/4)
    q3=np.sqrt((1-r11-r22+r33)/4)
    
    if (r32-r23)<0:
        q1=-q1
    if (r13-r31)<0:
        q2=-q2
    if (r21-r12)<0:
        q3=-q3
        
    q=np.array([q1,q2,q3])
    
    normq=np.linalg.norm(q)
    theta=2*np.arctan2(normq,q0)
    
    r = theta*q/normq
    
    return r

    
    
# Q5.3
# we're using numerical gradients here
# but one of the great things about the above formulation
# is it has a nice form of analytical gradient  计算残差
def rodriguesResidual(K1, M1, p1, K2, p2, x):   #p1和p2都包含很多点
    P,R,t=mystract(x) #提取参数
    M2=np.hstack((R,t[:,None]))
    
    num=p1.shape[0]  #个数
    
    C1=K1.dot(M1)
    C2=K2.dot(M2)
    
    err=np.zeros(num)
    
    for k in range(num):
        
        point=np.append(P[k,:],1)
        
        pt1=p1[k,:]
        pt2=p2[k,:] #找到两个点
        
        rept1=C1.dot(point)
        rept1=rept1/rept1[-1]
        rept1=rept1[0:-1]

        rept2=C2.dot(point)
        rept2=rept2/rept2[-1]
        rept2=rept2[0:-1]

        err[k]=np.linalg.norm(pt1-rept1)**2+np.linalg.norm(pt2-rept2)**2
        
    
    residuals = np.sum(err)
    
    return residuals

    
    
# we should use scipy.optimize.minimize
# L-BFGS-B is good, feel free to use others and report results
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
# 找到最佳的参数
def bundleAdjustment(K1, M1, p1, K2, M2init, p2, Pinit):  #Pinit是矩阵的形式
    R=M2init[:,0:-1]
    t=M2init[:,-1]

#K1,C1,goodP1,K2,C2,goodP2,P

    xinit=mycompact(Pinit,R,t)
    
    additional=(K1, M1, p1, K2, p2)  #额外的参数
    
    res = scipy.optimize.minimize(fun=myfun, args=additional, x0=xinit)
    
    bestx=res['x']
    
    P,R,t = mystract(bestx)
    M2=np.hstack((R,t[:,None]))
    
    return M2,P 
    
   
    
#自定义的函数    
def mycalc(pts1,pts2,F):  #计算两点在什么程序上符合F
    num=pts1.shape[0] #个数    
    err=np.zeros(num)
    
    for k in range(num):
        P1=np.append(pts1[k],1)
        P2=np.append(pts2[k],1)
        
        err[k]=P2.dot(F.dot(P1)) 
    
    return err
    
def mycompact(P,R,t):#转化为一个向量
    x1=P.flatten()
    x2=R.flatten()
    x3=t
    x=np.hstack((x1,x2,x3))
    return x
    
    
def mystract(x):#转化为一个向量
    t=x[-3:]
    R=x[-12:-3]
    P=x[0:-12]

    R=np.reshape(R,(3,3))
    P=np.reshape(P,(-1,3))
    
    
    
    return P,R,t
    
    
def myfun(x, *args): #单参数的函数
    
    K1, M1, p1, K2, p2=args    #相关参数
    
    value=rodriguesResidual(K1, M1, p1, K2, p2, x)
    
    return value #计算返回值
    
    
    
    
    
    