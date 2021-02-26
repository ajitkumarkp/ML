import dlib,cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import faceBlendCommon as fbc

matplotlib.rcParams['figure.figsize'] = (15.0,15.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'bilinear'

# Landmark model
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Get the face detector object
faceDetector = dlib.get_frontal_face_detector()
# Get the landmark detector object
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

# face1 is the image of the girl w/o makeup
face1 = cv2.imread("girl-no-makeup.jpg")
# convert to RGB and preserve a copy for later use 
face1 = cv2.cvtColor(face1,cv2.COLOR_BGR2RGB)
face1_cp = np.copy(face1)

# face2 is the image of the girl w/ makeup
face2 = cv2.imread("girl-blush.jpg")
# Resize face2 to the shape of face1 and convert to RGB 
face2 = cv2.resize(face2, (face1.shape[1], face1.shape[0]))
face2 = cv2.cvtColor(face2,cv2.COLOR_BGR2RGB)

# Get all 68 facepoints for both the images
face1_points = fbc.getLandmarks(faceDetector, landmarkDetector, face1)
face2_points = fbc.getLandmarks(faceDetector, landmarkDetector, face2)

# Based on the Dlib face points template, identify 4 points around the right and left cheeks.
# The area inside these 4 points is where the blush will be applied. 
face1_Rcheek = (face1_points[35], face1_points[42], face1_points[15],face1_points[12])
face1_Lcheek = (face1_points[31], face1_points[39], face1_points[1], face1_points[4])

# Right cheek area is divided into 2 triangles i.e upper right and lower right 
# Warp upper right triangle from face2 to face1
face1_tU_R = (face1_points[35], face1_points[42], face1_points[15])
face2_tU_R = (face2_points[35], face2_points[42], face2_points[15])
fbc.warpTriangle(face2, face1, face2_tU_R, face1_tU_R)
# Warp lower right triangle from face2 to face1
face1_tL_R = (face1_points[35], face1_points[15], face1_points[12])
face2_tL_R = (face2_points[35], face2_points[15], face2_points[12])
fbc.warpTriangle(face2, face1, face2_tL_R, face1_tL_R)

# Similarly left cheek area is divided into 2 triangles i.e upper left and lower left  
# Warp upper left triangle from face2 to face1
face1_tU_L = (face1_points[31], face1_points[39], face1_points[1])
face2_tU_L = (face2_points[31], face2_points[39], face2_points[1])
fbc.warpTriangle(face2, face1, face2_tU_L, face1_tU_L)
# Warp lower left triangle from face2 to face1
face1_tL_L = (face1_points[31], face1_points[1], face1_points[4])
face2_tL_L = (face2_points[31], face2_points[1], face2_points[4])
fbc.warpTriangle(face2, face1, face2_tL_L, face1_tL_L)

# Display the warped cheeks image
# plt.title("face1- Cheeks warped");plt.imshow(face1);plt.show()

##################
## Adding Blush ##
##################

# 1. To the Right Cheek
# 1.a Find convex hull for the right cheek
hull1_R = []
hullIndex = cv2.convexHull(np.array(face1_Rcheek), returnPoints=False)
for i in range(0, len(hullIndex)):
    hull1_R.append(face1_Rcheek[hullIndex[i][0]])

# 1.b Create a Mask for Seamless cloning the area around right cheek
mask = np.zeros(face1.shape, dtype=face1.dtype)  
cv2.fillConvexPoly(mask, np.int32(hull1_R), (255, 255, 255))
r = cv2.boundingRect(np.float32([hull1_R]))    
center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

# 1.c Seamless cloning.
# face1 is the src image containing warped cheeks
# face1_cp is the destination which is the original image w/o makeup 
face1_R = cv2.seamlessClone(np.uint8(face1), face1_cp, mask, center, cv2.NORMAL_CLONE)

# 2. Adding blush to the left cheek (Repeat the above ops for Left Cheek)
# 2.a Find convex hull for Left cheek
hull1_L = []
hullIndex = cv2.convexHull(np.array(face1_Lcheek), returnPoints=False)
for i in range(0, len(hullIndex)):
    hull1_L.append(face1_Lcheek[hullIndex[i][0]])

# 2.b Create a Mask for Seamless cloning the area around left cheek
mask = np.zeros(face1.shape, dtype=face1.dtype)  
cv2.fillConvexPoly(mask, np.int32(hull1_L), (255, 255, 255))
r = cv2.boundingRect(np.float32([hull1_L]))    
center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

# 2.c Seamless cloning.
# face1 is the src image containing warped cheeks
# face1_R is the destination which contains seamlessly cloned right cheek  
face1_blush = cv2.seamlessClone(np.uint8(face1), face1_R, mask, center, 
                          cv2.MIXED_CLONE)

# plt.title("With Blush Only"); plt.imshow(face1_blush); plt.show()

#####################
## Adding Earrings ##
#####################
# Load a cropped image of an earring and convert it to RGB  
earring = cv2.imread("earring.jpg")
earring = cv2.cvtColor(earring,cv2.COLOR_BGR2RGB)

# These are the coordinates of the left and right ears
face1_L_ear = face1_points[2]
face1_R_ear = face1_points[14]

# Resize the earring to a rough 40x120 pixels 
earring = cv2.resize(earring,(40, 120))
# Create a white mask of the shape of earring
mask = 255* np.ones(earring.shape, dtype=earring.dtype)  

# By trial n error determine approx points around the left 
# and right ear's coordinates to place the earrings
center_R = (face1_R_ear[0] + 15, face1_R_ear[1] + 25)
center_L = (face1_L_ear[0] - 3, face1_R_ear[1] + 25)

# Perform seamless Cloning to place the earrings on the 2 approx pts
face1_final = cv2.seamlessClone(earring, face1_blush, mask, center_R, cv2.MIXED_CLONE)
face1_final = cv2.seamlessClone(earring, face1_final, mask, center_L, cv2.MIXED_CLONE)

# plt.title("With Blush and Earrings"); plt.imshow(face1_final); plt.show()

#####################
## Adding Glasses  ##
#####################
# Load a cropped image of glasses and convert it to RGB  
glasses = cv2.imread("glasses.jpg")
glasses = cv2.cvtColor(glasses,cv2.COLOR_BGR2RGB)

# Determine the 4 corners of the rectangular region around the eyes where glasses will be placed
# i.e horizontally b/w face point #1 to #15 and vertically b/w #24 to #15 or #1
y = face1_points[24][1]
xr = face1_points[15][0]
xl = face1_points[1][0]
face1_Eye_pts = (face1_points[1], face1_points[15], (xr,y), (xl,y))

# Using the 4 cornor pts, crop out the eye region from the face
face1_Eye_ROI = face1_final[y:face1_points[1][1], face1_points[1][0]:face1_points[15][0]]

# Use face point #27 as the approx center to allign the glasses on the face
x = face1_points[27][0]
y = face1_points[27][1]
center = (x+10, y+10)

# Create a white mask for cloning
mask = 255* np.ones(glasses.shape, dtype=glasses.dtype)  

# Perform Seamless cloning of the glasses onto the face image with blush and earrings
# face1_goggles_normal = cv2.seamlessClone(glasses, face1_final, mask, center, cv2.NORMAL_CLONE)
face1_glasses_mixed = cv2.seamlessClone(glasses, face1_final, mask, center, cv2.MIXED_CLONE)

cv2.imwrite("Final.jpg", face1_glasses_mixed[:,:,::-1])

plt.subplot(121); plt.title("Original Face");  plt.imshow(face1_cp)
plt.subplot(122); plt.title("Face With Virtual makeup"); plt.imshow(face1_glasses_mixed)
plt.show()


