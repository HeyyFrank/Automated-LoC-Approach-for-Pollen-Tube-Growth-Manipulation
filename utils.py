# -*- coding: utf-8 -*-
"""
Useful functons.
"""
import numpy as np
import math
from statistics import mean
import matplotlib.pyplot as plt
import cv2
import sys
import time
import copy
import random as rng
import tkinter as tk
from PIL import Image, ImageTk


def gaussian(img, ksize=5):
    return cv2.GaussianBlur(img,(ksize,ksize),0)
    
def median(img, ksize=5):
    return cv2.medianBlur(img, ksize)

def img_process(frame, size=5):
    # frame = gaussian(frame, 9)
    # frame = median(frame, 9)
    # frame = gaussian(frame, 7)
    # frame = median(frame, 7)
    frame = gaussian(frame, size)
    frame = median(frame, size)
    return frame

def coord_trans(lines, p1):
    """ transform list of lines to a translated coord """
    for idx,line in enumerate(lines):
        offset = p1[0]*np.cos(line[1]) + p1[1]*np.sin(line[1])
        lines[idx, 0] += offset
    return lines
    
def distance(lines, center):
    lines_t = coord_trans(lines.copy(), -center)
    # d = lines_t[:,0].min(axis=0)
    # index = lines_t[:,0].argmin()
    return abs(lines_t[:,0])
    
def pt_theta_dist(p1, theta):
    d = p1[0]*np.cos(theta)+p1[1]*np.sin(theta)
    return d

def intersect(l1,l2):
    a1 = np.cos(l1[0,1])
    b1 = np.sin(l1[0,1])
    a2 = np.cos(l2[0,1])
    b2 = np.sin(l2[0,1])
    if b1 == 0:
        x = l1[0,0]*a1
        y = (l2[0,0]-x*a2)/b2
    elif b2 == 0:
        x = l2[0,0]*a2
        y = (l1[0,0]-x*a1)/b1
    else:
        x = (l2[0,0]/b2-l1[0,0]/b1)/(a2/b2-a1/b1)
        y = (l1[0,0]-x*a1)/b1
    return np.array((x,y))

def drawhoughLinesOnImage(image, houghLines, center=None, gamma=None, localLines=None):
    
    for idx,line in enumerate(houghLines):
        rho,theta = line[0],line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 4000*(-b))
        y1 = int(y0 + 4000*(a))
        x2 = int(x0 - 4000*(-b))
        y2 = int(y0 - 4000*(a))
        if idx==0 or idx==houghLines.shape[0]-1:
            color = 255 #50
        else:
            color = 255
        if localLines is not None:
            sect = intersect(houghLines, localLines).astype(int)
            p1=np.array((x1,y1))
            if (p1-sect)@gamma<0:
                p1=(x2,y2)
            cv2.line(image,tuple(sect),tuple(p1), color, 2) 
            for line1 in localLines:
                rho1,theta1 = line1[0],line1[1]
                a1 = np.cos(theta1)
                b1 = np.sin(theta1)
                x0l = a1*rho1
                y0l = b1*rho1
                x1l = int(x0l + 1000*(-b1))
                y1l = int(y0l + 1000*(a1))
                x2l = int(x0l - 1000*(-b1))
                y2l = int(y0l - 1000*(a1))
                p1l=np.array((x1l,y1l))
                if (p1l-sect)@gamma>0:
                    p1l=(x2l,y2l)
                cv2.line(image, tuple(sect), tuple(p1l), color, 2) 
        elif center is None:
            p1 = (x1,y1)
            p2 = (x2,y2)
            cv2.line(image,p1,p2, color, 2) 
        else:
            alpha = np.arctan2(gamma[1],gamma[0])
            connectLine = np.zeros((1,2))
            connectLine[0,0] = center[0]*np.cos(alpha)+center[1]*np.sin(alpha)
            connectLine[0,1] = alpha
            sect0 = intersect(houghLines, connectLine).astype(int)
            p1 = sect0
            p2 = np.array((x2,y2))
            if (p2-sect0)@gamma<0:
                p2=(x1,y1)
            cv2.line(image,tuple(p1),tuple(p2), color, 2)     

def blend_images(image, final_image, alpha=0.7, beta=1., gamma=0.):
    return cv2.addWeighted(final_image, alpha, image, beta,gamma)

def el2center(el,gamma):
    d = mean(el[1])*0.5
    cen = el[0]
    return int(cen[0]+d*gamma[0]), int(cen[1]+d*gamma[1])

def test(img):
    h, w = img.shape
    img[h//2:h//2+20,:] = 0
    return img

def canny_t(gray, cann):
    """ perform canny edge detection and return a filled convex contour """
    edged=cv2.Canny(gray,cann[0],cann[1], L2gradient = True)
    # cv2.imshow( 'edged', cv2.pyrUp(edged),)
    
    # hull3 = cv2.findNonZero(edged)
    # filled3 = np.zeros_like(edged) 
    # cv2.fillPoly(filled3, pts =[hull3], color=255)
    # cv2.imshow( 'fill', filled3,)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # temp = np.zeros_like(gray)
    # for i in contours:
    #     cv2.drawContours(temp,[i],0,255,2)
    # cv2.imshow( 'cont', temp,)
    
    if contours:
        comb = np.vstack(contours)
        hull = cv2.convexHull(comb)
        filled = np.zeros_like(edged) 
        cv2.fillPoly(filled, pts =[hull], color=255)
        return filled
    else:
        return None


def thres_transform(gray, p1, cann=(80,180)):
    """ transform using thresholding, for fluorecent PT detection """
    # gray = test(gray.copy())
    _, th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,51,-2)
    th1 = median(th1, 9)
    th1 = cv2.dilate(th1,np.ones((5,5),np.uint8),iterations = 1)
    th2 = median(th2, 9)
    th2 = cv2.dilate(th2,np.ones((5,5),np.uint8),iterations = 1)
    # cv2.imshow( 'threshold2', th2,)
    cv2.imshow( 'threshold1', th1,)
    # edged=cv2.Canny(gray,cann[0],cann[1], L2gradient = True)
    th = cv2.findNonZero(th1)
    # cv2.imshow('canny edges',edged)
    _, contours, _ = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        comb = np.vstack(contours)
        hull = cv2.convexHull(comb)
        ellipse = list(cv2.fitEllipse(th))
        
        # test contour
        # drawing = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        # for i in range(1):
        #     color = (255, 255, 255)
        #     # cv2.drawContours(drawing, [comb], i, color)
        #     cv2.drawContours(drawing, [hull], i, color)
        # cv2.imshow('Contours', drawing)
    
        # process display in original image
        hull[:,:,0]+= p1[0]
        hull[:,:,1]+= p1[1]
        ellipse[0] = (ellipse[0][0]+p1[0], ellipse[0][1]+p1[1])
        ellipse = tuple(ellipse)
        # ellipse[0][0]+= p1[1]
        # ellipse[0][1]+= p1[0]
        return hull, ellipse
    else:
        return None, None
    
def update_bbox(el, center, bbox, gamma=None, roi=None):
    """ update bbox based on ellipse. """
    # determin long axis
    if el[1][0]>el[1][1]:
        long = 0
    else:
        long = 1
    e_cen = np.array(el[0])
    # x determines where to set roi center. x=2: end point of long axis
    x=2
    # determine moving direction if gamma=None
    d1 = [el[1][long]/x*(1-long), el[1][long]/x*long]
    d2 = [-el[1][long]/x*(1-long), -el[1][long]/x*long]
    d1 = np.array(rot(d1, el[2])) 
    d2 = np.array(rot(d2, el[2]))
    if gamma is None:
        d1 = d1 + e_cen
        d2 = d2 + e_cen
        dist1 = np.linalg.norm(d1-center)
        dist2 = np.linalg.norm(d2-center)
        if dist1 < dist2:
            dit = 1
        else:
            dit = -1
    
    elif np.dot(d1, gamma)>0:
        dit = 1
    else:
        dit = -1
    
    # set bbox center to 1/4 of the ellipse
    dt0 = [el[1][long]/x*(1-long), el[1][long]/x*long]
    dt0 = np.array(rot(dt0, el[2]))*dit
    dt = (dt0 + e_cen).astype(np.uint16)
    dt = dt - center
    if roi is None:
        bbox_ = list(bbox)
        bbox_[0] += dt[0]
        bbox_[1] += dt[1]
        bbox_ = tuple(bbox_)
        return bbox_, dt0/np.linalg.norm(dt0), np.linalg.norm(dt)
    else:
        roi[0]+=dt[0]
        roi[1]+=dt[0]
        roi[2]+=dt[1]
        roi[3]+=dt[1]
        return roi, dt0/np.linalg.norm(dt0), np.linalg.norm(dt)

def update_roi(roi, g):
    """ update roi based on movement vector """
    dx = g[0]
    dy = g[1]
    # transfer float to int
    dx=round(dx)
    dy=round(dy)
    roi[0]+=dx
    roi[1]+=dx
    roi[2]+=dy
    roi[3]+=dy
    return roi

def rot(p, theta):
    """ return pt after rotation of theta(in degrees).
    p: pt coordinate, [x,y] """
    theta = theta/180*np.pi
    return [p[0]*np.cos(theta) - p[1]*np.sin(theta), 
            p[1]*np.cos(theta) + p[0]*np.sin(theta)]
    
def bbox2pt(bbox):
    """ transfer bbox to points and float -> int. """
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    return p1, p2

def roi2pt(roi):
    """ transfer bbox to points and float -> int. """
    p1 = (int(roi[0]), int(roi[2]))
    p2 = (int(roi[1]), int(roi[3]))
    return p1, p2

def blend_box(b1, b2, alp=.5, beta=.5):
    bbox = list(b1)
    bbox[0] = alp*b1[0] + beta*b2[0]
    bbox[1] = alp*b1[1] + beta*b2[1]
    return tuple(bbox)

def blend_ellip(el1, el2, alp=.5, beta=.5):
    el = list(el1)
    ok = 1
    move = np.linalg.norm(np.array(el1[0])-np.array(el2[0]))
    if move> np.mean(np.array(el1[1]))*1.:
        ok = 0
    # location
    el[0] = (alp*el1[0][0] + beta*el2[0][0], alp*el1[0][1] + beta*el2[0][1])
    # size
    alp = 0.1
    beta = 0.9
    el[1] = (alp*el1[1][0] + beta*el2[1][0], alp*el1[1][1] + beta*el2[1][1])
    # angle
    el[2] = 0.4*el1[2] + 0.6*el2[2]
    return tuple(el), ok

def inert_ellip(el, gamma, dist=0):
    """ move the ellipse a little bit towards the moving direction to compensate detection error.
    gamma: moving directional unit vector. """
    el = list(el)
    cen = np.array(el[0])
    el[0] = tuple(cen + gamma*dist)
    # el[0] = (el[0][0] + gamma*dist, el[0][1] + gamma*el[0][1])
    return tuple(el)

def blend_lines(l1, l2, alp=.5, beta=.5):
    # lines = []
    # for a,b in zip(l1,l2):
    #     l=[]
    #     for i in len(a):
    #         l.append( alp*a[i] + beta*b[i] )
        
    #     lines.append(l)
    return l1*alp+l2*beta

def circular_mask(temp, gamma=None, ratio=0.9):
    h,w = temp.shape
    # construct a circular mask
    xc = w//2
    yc = h//2
    mask = np.zeros_like(temp).astype(np.uint8)
    mask = cv2.circle(mask, (xc,yc), int(min(xc, yc)*ratio), 255, -1)
    # filter out tip part
    if gamma is not None:
        d1=np.array([1,0])
        d2=np.array([-1,0])
        d3=np.array([0,1])
        d4=np.array([0,-1])
        if np.dot(d1, gamma)>0.7071:
            mask[:,xc:]=255
        elif np.dot(d2, gamma)>0.7071:
            mask[:,:xc]=255
        elif np.dot(d3, gamma)>0.7071:
            mask[yc:,:]=255
        elif np.dot(d4, gamma)>0.7071:
            mask[:yc,:]=255
    # cv2.imshow('mask', mask)                
    return mask

def ring_mask(temp, cen, r, w):
    """ Ring mask for centerline determination 
    Args:
        cen[1x2 array]: center of the ring
        r [float]: radius
        w [float]: width of ring
    """
    # construct a circular mask
    xc = int(cen[0])
    yc = int(cen[1])
    mask = np.zeros_like(temp).astype(np.uint8)
    mask = cv2.circle(mask, (xc,yc), int(r), 255, w)
    # mask = cv2.circle(mask, (xc,yc), int(r-w/2), 0, -1)    
    return mask

def outer_mask(temp, cen, r):
    """ Outer mask for centerline determination 
    Args:
        cen[1x2 array]: center of the ring
        r [float]: radius
    """
    # construct a circular mask
    xc = int(cen[0])
    yc = int(cen[1])
    mask = np.ones_like(temp).astype(np.uint8)*255
    mask = cv2.circle(mask, (xc,yc), int(r), 0, -1)
    return mask

def get_centroid(c):
    """
    Compute the centroid of contour c
    Args:
        c: contour as a 1x2xN numpy array
    Returns:
        (cx,cy): tuple of the centroid position 
    """

    mom = cv2.moments(c)
    cx = int(mom['m10']/mom['m00'])
    cy = int(mom['m01']/mom['m00'])
    return (cx, cy)

def dbscan_predict(model, X):
    nr_samples = 1
    y_new = -1
    # y_new = np.ones(shape=nr_samples, dtype=int) * -1

    for i in range(nr_samples):
        diff = model.components_ - X  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new = model.labels_[model.core_sample_indices_[shortest_dist_idx]]
    return y_new  
      
def check_val(el, bp):
    pass
    
    
class MousePositionTracker(tk.Frame):
    """ Tkinter Canvas mouse position widget. """

    def __init__(self, canvas):
        self.canvas = canvas
        self.enable = True
        self.canv_width = self.canvas.cget('width')
        self.canv_height = self.canvas.cget('height')
        self.reset()

        # Create canvas cross-hair lines.
        xhair_opts = dict(dash=(3, 2), fill='white', state=tk.HIDDEN)
        self.lines = (self.canvas.create_line(0, 0, 0, self.canv_height, **xhair_opts),
                      self.canvas.create_line(0, 0, self.canv_width,  0, **xhair_opts))

    def cur_selection(self):
        return (self.start, self.end)

    def begin(self, event):
        self.hide()
        self.start = (event.x, event.y)  # Remember position (no drawing).

    def update(self, event):
        self.end = (event.x, event.y)
        self._update(event)
        self._command(self.start, (event.x, event.y))  # User callback.

    def _update(self, event):
        # Update cross-hair lines.
        self.canvas.coords(self.lines[0], event.x, 0, event.x, self.canv_height)
        self.canvas.coords(self.lines[1], 0, event.y, self.canv_width, event.y)
        self.show()

    def reset(self):
        self.start = self.end = None

    def hide(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.HIDDEN)
        self.canvas.itemconfigure(self.lines[1], state=tk.HIDDEN)

    def show(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.NORMAL)
        self.canvas.itemconfigure(self.lines[1], state=tk.NORMAL)

    def autodraw(self, command=lambda *args: None):
        """Setup automatic drawing; supports command option"""
        if self.enable:
            self.reset()
            self._command = command
            self.canvas.bind("<Button-1>", self.begin)
            self.canvas.bind("<B1-Motion>", self.update)
            self.canvas.bind("<ButtonRelease-1>", self._quit)

    def _quit(self, event):
        self.hide()  # Hide cross-hairs.
        self.reset()


class SelectionObject:
    """ Widget to display a rectangular area on given canvas defined by two points
        representing its diagonal.
    """
    def __init__(self, canvas, select_opts):
        # Create attributes needed to display selection.
        self.canvas = canvas
        self.select_opts1 = select_opts
        self.width = int(self.canvas.cget('width'))
        self.height = int(self.canvas.cget('height'))

        # Options for areas outside rectanglar selection.
        select_opts1 = self.select_opts1.copy()  # Avoid modifying passed argument.
        select_opts1.update(state=tk.HIDDEN)  # Hide initially.
        # Separate options for area inside rectanglar selection.
        select_opts2 = dict(dash=(2, 2), fill='', outline='white', state=tk.HIDDEN)

        # Initial extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = 0, 0,  1, 1
        omin_x, omin_y,  omax_x, omax_y = 0, 0,  self.width, self.height
        self.bbox = None
        
        self.rects = (
            # Area *outside* selection (inner) rectangle.
            self.canvas.create_rectangle(omin_x, omin_y,  omax_x, imin_y, **select_opts1),
            self.canvas.create_rectangle(omin_x, imin_y,  imin_x, imax_y, **select_opts1),
            self.canvas.create_rectangle(imax_x, imin_y,  omax_x, imax_y, **select_opts1),
            self.canvas.create_rectangle(omin_x, imax_y,  omax_x, omax_y, **select_opts1),
            # Inner rectangle.
            self.canvas.create_rectangle(imin_x, imin_y,  imax_x, imax_y, **select_opts2)
        )

    def update(self, start, end):
        # Current extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = self._get_coords(start, end)
        omin_x, omin_y,  omax_x, omax_y = 0, 0,  self.width, self.height

        # Update coords of all rectangles based on these extrema.
        self.canvas.coords(self.rects[0], omin_x, omin_y,  omax_x, imin_y),
        self.canvas.coords(self.rects[1], omin_x, imin_y,  imin_x, imax_y),
        self.canvas.coords(self.rects[2], imax_x, imin_y,  omax_x, imax_y),
        self.canvas.coords(self.rects[3], omin_x, imax_y,  omax_x, omax_y),
        self.canvas.coords(self.rects[4], imin_x, imin_y,  imax_x, imax_y),

        for rect in self.rects:  # Make sure all are now visible.
            self.canvas.itemconfigure(rect, state=tk.NORMAL)
        self.roi = [ [imin_y,imin_x],  [ imax_y, imax_x]]
        self.bbox = (imin_x/self.width,imin_y/self.height, (imax_x-imin_x)/self.width, (imax_y-imin_y)/self.height)

    def _get_coords(self, start, end):
        """ Determine coords of a polygon defined by the start and
            end points one of the diagonals of a rectangular area.
        """
        return (min((start[0], end[0])), min((start[1], end[1])),
                max((start[0], end[0])), max((start[1], end[1])))

    def hide(self):
        for rect in self.rects:
            self.canvas.itemconfigure(rect, state=tk.HIDDEN)
            
    def get_roi(self):
        return self.bbox
