# -*- coding: utf-8 -*-
"""
Calculate magnetic field of straight current segments

Created on Fri Apr 24 18:45:58 2020

@author: Derick Canceran
@email: dbcanceran@gmail.com
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# MTQ dimensions
L = 0.5   # meters
W = 0.5   # meters

current = 0.5 # amperes
mu_0 = 4*np.pi*10**(-7)

# Define corners of MTQ, placed on XY plane
#coilCorners = np.array([[ L/2,  W/2, 0],
#                        [-L/2,  W/2, 0],
#                        [-L/2, -W/2, 0],
#                        [ L/2, -W/2, 0]])
coilCorners = np.load('mtqpoints.npy')
coilCorners /= 1000 # convert to meters
coilCorners = np.concatenate((coilCorners, np.zeros((len(coilCorners),1))), axis=1)

# Define space to calculate magnetic field
samp = 201
sampling = 0
x = np.linspace(0.1,0,samp)#np.linspace(0.05,-0.05,sampling)
X, Y, Z = np.meshgrid(x,x,np.arange(0,1),indexing='ij')
A = np.stack((X,Y,Z),axis=3)
totalField = np.zeros(A.shape)

# for each segment of the coil
for i in range(1,len(coilCorners)):
    # Calculate Magnetic field magnitude using the formula
    # B(r) = \frac{\mu I}{4\pi R} (\sin{\theta2} - \sin{theta1})
    #       ---                 x
    #        ^                / : \
    #        |              /<--:-->\
    #      R |            /  t1 : t2  \
    #        |          /       :       \
    #        |        /         :         \
    #       ---      -=====================-
    
    # Calculate distance from segment axis
    segmentVector = coilCorners[i] - coilCorners[i-1]
    segmentVectorUnit = segmentVector/np.sqrt((segmentVector**2).sum())
    startToPointInSpace = A - coilCorners[i-1]
    dot = np.dot(startToPointInSpace,segmentVectorUnit)
    projection = np.multiply.outer(dot,segmentVectorUnit)
    projection += coilCorners[i-1]
    fromPointToProjectionVector = projection-A
    distance = np.sqrt(((fromPointToProjectionVector)**2).sum(axis=3))
    
    # Determine if projection of point is space is within the length
    # of current segment. This determines the sign of the second term
    # in the formula. (When negative, the projection point is within 
    # the length of the segment, and each portion of the segment
    # contributes to the filed, i.e. \sin{\theta1}<0. When positive, the
    # projection point is outside the length of the segment, \sin{\theta1}>0,
    # effectively removing the ``phantom'' portion from the 
    # ``extended'' segment.)
    #
    # The segment is parametrized by variable `t`. Reference point (where t=0)
    # is taken to be at the middle of the segment. So, checking if the
    # projection point is within the segment is simply knowing if 
    # abs(t)<0.5 (here, abs(t)>0.5 is used to know which are NOT contained
    # along the length of the segment.)
    for j in range(3):          # Find non-zero reference
        if abs(segmentVector[j])>1e-10:
            break
    t = ( (projection[:,:,:,j] - (coilCorners[i-1,j]+coilCorners[i,j])/2)/
          segmentVector[j] )
    t = 2*(np.abs(t)>0.5) - 1       # Make +1 and -1 factors
    
    # Calculate sin theta1
    fromPointToProjectionVectorUnit = ( fromPointToProjectionVector/
                                        np.multiply.outer(distance,[1,1,1]) )
    initVec = -startToPointInSpace
    vecMag = np.sqrt((initVec**2).sum(axis=3))
    initVecUnit = initVec/np.multiply.outer(vecMag,[1,1,1])
    cross = np.cross(fromPointToProjectionVectorUnit, initVecUnit)
    sinTheta1 = np.sqrt((cross**2).sum(axis=3))
    
    # Calculate sin theta2
    finVec = coilCorners[i] - A
    vecMag = np.sqrt((finVec**2).sum(axis=3))
    finVecUnit = finVec/np.multiply.outer(vecMag,[1,1,1])
    cross = np.cross(fromPointToProjectionVectorUnit, finVecUnit)
    sinTheta2 = np.sqrt((cross**2).sum(axis=3))
    
    
    magFieldMagnitude = ( mu_0*current*np.abs(sinTheta2 - t*sinTheta1)/
                          (4*np.pi*distance) )
    
    # Calculate direction of field
    magFieldDirection = np.cross(fromPointToProjectionVectorUnit,
                                 segmentVectorUnit)
    
    # Calculate magnetic field
    magField = np.multiply.outer(magFieldMagnitude,[1,1,1])*magFieldDirection
    
    totalField += magField

totalFieldMagnitude = np.sqrt((totalField**2).sum(axis=3))
totalFieldDirection = totalField/np.multiply.outer(totalFieldMagnitude,[1,1,1])
#plt.quiver(A[:,sampling//2,:,2],
#           A[:,sampling//2,:,0],
#           totalFieldDirection[:,sampling//2,:,2],
#           totalFieldDirection[:,sampling//2,:,0])
#plt.gca().set_aspect('equal')
plt.imshow(totalFieldMagnitude[:,:,sampling//2])