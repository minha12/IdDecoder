#! /usr/bin/env python

import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt

class Face:
    def __init__(self,img, img_path=None, predictor_path='./pretrained_models/shape_predictor_68_face_landmarks.dat'
                ):
        self.img_path = img_path
        self.predictor_path = predictor_path
        if img_path==None:
            if isinstance(img,np.ndarray)==False:
                self.face = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            else:
                self.face = img
        else:
            self.face = cv2.imread(self.img_path)
        self.gray = cv2.cvtColor(self.face, cv2.COLOR_BGR2GRAY)
        self.height, self.width, self.channels = self.face.shape
        self.landmarks = []
        self.rect = []
        self.convexHull = []
        self.triangles = []
    
    def get_landmarks(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.predictor_path)
        self.rect = detector(self.gray)[0]
        landmarks = predictor(self.gray, self.rect)
        landmarks_points = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x,y))
        self.landmarks = np.array(landmarks_points, np.int32)
        self.convexHull = cv2.convexHull(self.landmarks)
        return self.landmarks
    
    def get_triangles(self):
        if len(self.convexHull)==0:
            raise Exception('Call get_landmarks() first!')
        rect_landmarks = cv2.boundingRect(self.convexHull)
        #calculate the Delaunay Triangulation
        subdiv = cv2.Subdiv2D(rect_landmarks)
        subdiv.insert(self.landmarks.tolist())
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)
        self.triangles = triangles
        return self.triangles
        
    def indexed_triangles(self):
        triangles = self.get_triangles()
        indexes_triangles = []
        for triangle in triangles:
            pt1 = (triangle[0], triangle[1])
            pt2 = (triangle[2], triangle[3])
            pt3 = (triangle[4], triangle[5])
            
            index_pt1 = np.where((self.landmarks == pt1).all(axis=1))
            index_pt1 = get_index(index_pt1)
            index_pt2 = np.where((self.landmarks == pt2).all(axis=1))
            index_pt2 = get_index(index_pt2)
            index_pt3 = np.where((self.landmarks == pt3).all(axis=1))
            index_pt3 = get_index(index_pt3)
            
            # Saves coordinates if the triangle exists and has 3 vertices
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                vertices = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(vertices)
        return indexes_triangles

    def inverted_mask(self):
        init_src_mask = np.zeros_like(self.gray)
        self.get_landmarks()
        src_mask = cv2.fillConvexPoly(init_src_mask, self.convexHull, 255)
        return cv2.bitwise_not(src_mask)
    
    def show_convex_hull(self):
        #self.face.get_convexHull()
        face_cp = self.face.copy()
        face_applied_convex = cv2.polylines(face_cp, 
                                            pts=[self.convexHull], 
                                            isClosed=True, 
                                            color=(255,255,255), 
                                            thickness=3)
        #plt.imshow(cv2.cvtColor((face_applied_convex),cv2.COLOR_BGR2RGB))
        #plt.show()
        cv2.imshow("Output", face_applied_convex)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class FaceSwap():
    def __init__(self,src, tar,shape_predictor_path = './pretrained_models/shape_predictor_68_face_landmarks.dat'):
        self.src_face = Face(src)
        self.src_face.get_landmarks()
        self.tar_face = Face(tar)
        self.tar_face.get_landmarks()
        
    def warp_face(self):
        warpped_result = np.zeros(self.src_face.face.shape, np.uint8)
        tar_triangles = self.tar_face.indexed_triangles()
        for triangle in tar_triangles:
            _, vertices_tar, rect_tar = crop2_triangle(triangle, 
                                                       self.tar_face.landmarks)
            (x_0,y_0,width_0,height_0) = rect_tar
            cropped_tar = self.tar_face.face[y_0: y_0+height_0, x_0: x_0+width_0]

            cropped_mask, vertices_src, rect_src = crop2_triangle(triangle, 
                                                   self.src_face.landmarks)
            (x,y,width,height) = rect_src

            M = cv2.getAffineTransform(vertices_tar, vertices_src)  
            warpped_tar = cv2.warpAffine(cropped_tar, M, (width, height))
            warpped_tar = cv2.bitwise_and(warpped_tar, warpped_tar, mask=cropped_mask)

            new_rect = warpped_result[y: y+height, x: x+width]
            new_rect_gray = cv2.cvtColor(new_rect, cv2.COLOR_BGR2GRAY)

            masked_rect = cv2.threshold(new_rect_gray, 0, 255, 
                                        cv2.THRESH_BINARY_INV)
            warpped_tar = cv2.bitwise_and(warpped_tar, 
                                          warpped_tar, mask=masked_rect[1])

            new_rect = cv2.add(new_rect, warpped_tar)
            warpped_result[y: y+height, x: x+width] = new_rect
        return warpped_result
    
    def inverted_mask(self):
        init_src_mask = np.zeros_like(self.src_face.gray)
        src_mask = cv2.fillConvexPoly(init_src_mask, self.src_face.convexHull, 255)
        mask = cv2.bitwise_not(src_mask)
        mask = cv2.GaussianBlur( mask, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
        return mask
    
    def swap_face(self):
        warpped_tar = self.warp_face()
        init_src_mask = np.zeros_like(self.src_face.gray)
        src_mask = cv2.fillConvexPoly(init_src_mask, self.src_face.convexHull, 255)
        
        #src_mask_blur= cv2.GaussianBlur( src_mask, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
        #src_mask = cv2.addWeighted( src_mask_blur, 1.5, src_mask, -0.5, 0)
        src_mask_invert = cv2.bitwise_not(src_mask)
        
        src_faceless = cv2.bitwise_and(self.src_face.face, 
                                       self.src_face.face,
                                       mask=src_mask_invert
                                      )
        swapped_face = cv2.add(src_faceless,warpped_tar)
        
        # seamlessClone
        (x, y, widht, height) = cv2.boundingRect(self.src_face.convexHull)
        src_center = (int((x+x+widht)/2), int((y+y+height)/2))
        swapped_face = cv2.seamlessClone(swapped_face, 
                                          self.src_face.face, 
                                          src_mask, 
                                          src_center, 
                                          cv2.NORMAL_CLONE)
        return swapped_face
        
def get_index(arr):
    index = 0
    if not len(arr[0])==0:
        index = arr[0][0]
    return index

def crop2_triangle(triangle, landmarks):
    points = []
    for i in range(0,3):
        points.append(landmarks[triangle[i]])

    # Get bounding rect of the triangle
    rect = (x, y, width, height) = cv2.boundingRect(np.array(points, np.int32))
    
    mask_of_rect = np.zeros((height, width), np.uint8)

    # Moving the coordinates triangles vertices to the cropped rect
    new_vertices = np.array([[points[0][0]-x, points[0][1]-y], 
                             [points[1][0]-x, points[1][1]-y], 
                             [points[2][0]-x, points[2][1]-y]], np.int32)
    cv2.fillConvexPoly(mask_of_rect, new_vertices, 255)
    return mask_of_rect, np.float32(new_vertices), rect
        
if __name__ == '__main__':
    #test
    shape_predictor = 'shape_predictor_68_face_landmarks'
    src = 'trump.jpeg'
    tar = 'obama.jpg'
    fs = FaceSwap(src,tar,shape_predictor)
    swap = fs.swap_face()
    cv2.imshow("Face Swapped", swap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    
    
        
        

