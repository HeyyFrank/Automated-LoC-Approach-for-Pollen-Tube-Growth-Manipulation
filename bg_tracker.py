# -*- coding: utf-8 -*-
"""
Background subtraction Tracker.
"""
import numpy as np
from sklearn.cluster import KMeans,DBSCAN
from sklearn.mixture import GaussianMixture
import math
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys
import time
import copy
import random as rng
from Tracker import Tracker
from utils import *
import traceback

class bg_Tracker(Tracker):
    def __init__(self, img_width, img_height):
        """
        Initialize the background subtractor tracker
        Args:
            roi: region of interest, format is a numpy array defined as
                 [left_col, right_col, top_row, bottom_row]
            img_width: width of image (default 1920)
            img_height: height of image (default 1200)
            SNR_level: const for auto denoising, 2 for SNR & 1 for interface size
        """
        self.bg_tracker = cv2.createBackgroundSubtractorKNN(detectShadows =False)
        self.image_width = img_width
        self.image_height = img_height
        self.SNR_level = [0.82, 0.70, 150]
             
    def init_tracker(self, idx, frame, bbox, stride, BF=1, bg=1, process=True):
        self.st = idx
        self.last_frame = frame.copy()
        self.sd = stride
        self.BF = BF
        
        roi = np.array(bbox).reshape((2,2))
        roi[1,:] = roi[1,:]+roi[0,:]
        roi=roi.transpose()
        roi=roi.reshape(4)

        if roi is not None:
            self.roi = roi
        else:
            print('No region of interest set - setting default values [0 1919 0 1079]')
            self.roi = np.zeros(4,dtype=int)
            self.roi[1] = self.image_width-1
            self.roi[3] = self.image_height-1

        self.data = dict()
        self.bp_ellip = None
        self.bp_lines = None
        self.center = np.array([bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2])
        self.gamma = None
        self.walls = {'angle':[], 'wall1':[], 'wall2':[], 'len':10}
        self.data[idx] = {'center': self.center.copy(), 'ellipse': None,
                          'roi': roi.copy(), 'gamma': None, 'frame': self.last_frame.copy()}
        self.idx_list=[]
        self.center_list = []
        self.mot_traj = np.zeros_like(frame).astype(bool)
        self.fluo_traj = np.zeros_like(frame).astype(bool)
        self.last_bg_frame = {}
        self.el_track = False
        if BF:
            self.track(idx, frame, False, None, bg)
        else:
            self.track(idx, frame, False, None, 2)
        return True
        
    def find_bg_tip(self, img, eps=10, msp=15, w_area=0, fluo=0):
        """ 
        Cluster the motion subtraction mask with DBSCAN, and 
        return the clustered part (same size as img) that is closest to the center 
        """
        h,w = np.shape(img)
        cur_center = np.array([w//2, h//2])

        collec = cv2.findNonZero(img)
        collec = np.squeeze(collec)
        if len(collec.shape)<2:
            return img
        if fluo:
            # collec = np.concatenate((collec,[[w//2,h//2]]))
            try:
                centerline00 = np.array(self.data[self.idx_list[-2]]['centerline'][:3])
            except:
                centerline00 = np.array([[w//2, h//2]])
        # perform DBSCAN clustering (more robust to outliers)
        db = DBSCAN(eps=eps, min_samples=msp)
        db.fit(collec)
        labels = db.labels_
        mask = np.zeros_like(labels, dtype=bool)
        mask[db.core_sample_indices_] = True
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # if the mask is almost empty
        if n_clusters_<1:
            return img
        centers = np.zeros((n_clusters_, collec.shape[1]+1))
        for k in range(n_clusters_):
            member_mask = labels == k
            member = collec[mask & member_mask]
            centers[k,:-1] = np.mean(member, axis=0)
            k_center = np.array([centers[k,0],centers[k,1]])
            # calculate weights based on distance to roi center
            if fluo:
                dist = np.linalg.norm(centerline00 - k_center, axis = 1)
                centers[k,-1] = dist.mean()/max(img.shape)
            else:
                centers[k,-1] = np.linalg.norm(k_center-cur_center)/max(img.shape)
            # weight regarding number of points in each cluster
            if w_area>0:
                centers[k,-1] -= member.shape[0]/collec.shape[0]*w_area
        tmp = np.argmin(centers[:,-1])
        # centers = centers[centers[:,-1].argsort(), :]
        tip = np.zeros_like(img)
        member_mask = labels == tmp
        member = collec[mask & member_mask]
        for j in range(member.shape[0]):
            # cv2.circle(tip, (member[j,0], member[j,1]), radius=0, color=(255,255,255), thickness=-1)
            tip[ member[j,1], member[j,0]] = 255
        
        return tip
        
    def track(self, idx, img0, inter, ibox, bg=1, intf_ref=None, intf_reg='auto'):
        """
        Track blobs in image and return positions of centroids and image with tracked object
        Args:
            bg (int) : using fluo thresholding (2) or background subtraction tracking (1) or ellipse tracking (0)
            intf_ref (list): reference interface location ratio computed based on flow rates
            intf_reg (str): region for interfave detection, 'auto'(extend to PT) or 'fixed'(only loi)
        """
        self.idx_list.append(idx)
        self.intf_ref = intf_ref
        self.intf_reg = intf_reg
        # Start timer
        timer = cv2.getTickCount()
        # try:
        if bg==2:
            img, pos = self.fluo_track(idx, img0, inter, ibox)
        elif bg==1:
            img, pos = self.motion_track(idx, img0, inter, ibox)
        else:
            img = cv2.medianBlur(img0,5) 
            frame = img[self.roi[2]:self.roi[3],self.roi[0]:self.roi[1]] 
            img, pos = self.update(idx, img0, frame, 0, inter, ibox)
        # except:
        #     img = img0
        #     pos = []
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        cv2.putText(img, "FPS : " + str(int(fps)), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2);
        return img, pos
    
    def fluo_track(self, idx, img0, inter, ibox):
        """ Fluo thrsholding tracking """
        img = cv2.medianBlur(img0,5) 
        frame = img[self.roi[2]:self.roi[3],self.roi[0]:self.roi[1]] 
        # TODO: pyrDown
        # img_down = img0.copy()
        # img_down = cv2.pyrDown(img_down)
        # img_down = cv2.pyrDown(img_down)
        # img_down = img_process(img_down,3)
        # th0 = cv2.adaptiveThreshold(img_down, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #     cv2.THRESH_BINARY,25,-1)
        # th3 = cv2.pyrUp(th0)
        # th3 = cv2.pyrUp(th3)
        # th3 = th3[self.roi[2]:self.roi[3],self.roi[0]:self.roi[1]] 
        # # cv2.imshow( 'threshold 1', th3,)
        # ret,th3 = cv2.threshold(th3,150,255,cv2.THRESH_BINARY)
        
        # ret,th3 = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thv = np.average(frame)+np.std(frame)*0.8
        ret,th3 = cv2.threshold(frame,thv,255,cv2.THRESH_BINARY)
        # cv2.imshow('1.4',th3)
        # kernel = np.ones((5,5),np.uint8)
        # th3 = cv2.erode(th3,kernel,iterations = 2)
        # th3 = cv2.dilate(th3, kernel,iterations = 2)
        
        th3= cv2.morphologyEx(th3, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        # cv2.imshow('1',th3)
        pt0 = self.find_bg_tip(th3, eps=6, w_area=0.1, fluo=1)
        self.fluo_traj[self.roi[2]:self.roi[3],self.roi[0]:self.roi[1]] |= pt0.astype(bool)
        # contours, _ = cv2.findContours(pt, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        # cv2.imshow('pt',pt0)
        
        # try:
        #     tip = self.pre_tip
        #     num = cv2.findNonZero(tip).shape[0]
        #     while 1:
        #         tip = cv2.erode(tip,np.ones((3,3),np.uint8),iterations = 1)
        #         if cv2.findNonZero(tip).shape[0]<num*0.5:
        #             break
        #         pt = pt0 | tip
        # except:
        #     pt = pt0
        pt=pt0
        self.pre_tip = pt.copy()
        # cv2.imshow('pt merge',pt)
        
        # find centerline
        h,w = np.shape(frame)
        mid_point = []
        try:
            mask_edge = np.ones_like(pt)
            mask_edge[5:h-5,5:w-5] = 0
            pt_tmp = pt*mask_edge.astype(bool)
            M = cv2.moments(pt_tmp)
            rect = cv2.minAreaRect(cv2.findNonZero(pt_tmp))
            maxm = max(rect[1][0], rect[1][1])
        
            cX = (M["m10"] / M["m00"])
            cY = (M["m01"] / M["m00"])
            mid_point.append(np.array([cX, cY]))
        
            pt_tmp = pt*ring_mask(frame, mid_point[-1], maxm/2*1.2, 3).astype(bool)
            pt_next = pt*outer_mask(frame, mid_point[-1], maxm/2*1.5).astype(bool)
            pt_backup = pt_next
            M = cv2.moments(pt_tmp)
            maxm = 0
        except:
            pt01 = np.array([self.roi[0],self.roi[2]])
            roi00 = self.data[self.idx_list[-2]]['roi']
            pt00 = np.array([roi00[0],roi00[2]])
            centerline00 = self.data[self.idx_list[-2]]['centerline']
            centerline01 = centerline00 + pt00 - pt01

            rect = cv2.minAreaRect(cv2.findNonZero(pt))
            maxm = min(rect[1][0], rect[1][1])
            pt_next = pt
            for i in range(0,len(centerline01)-1):
                pt1 = centerline01[i]
                if 0<pt1[0]<w and 0<pt1[1]<h:
                    
                    mid_point.append(pt1)
                    pt_tmp = pt_next*ring_mask(frame, mid_point[-1], maxm/2*1.2, 3).astype(bool)
                    pt_tmp[:,:int(pt1[0])]=0
                    pt_next = pt_next*outer_mask(frame, mid_point[-1], maxm/2*1.5).astype(bool)
                    pt_next[:,:int(pt1[0])]=0    
                    break

            pt_backup = pt_next
            M = cv2.moments(pt_tmp)

        pt_backup1 = pt_backup
        while M["m00"] != 0 :
            rect = cv2.minAreaRect(cv2.findNonZero(pt_tmp))
            maxm1 = max(rect[1][0], rect[1][1])
            if maxm1 < maxm*0.5:
                break
            maxm = maxm1
            pt_backup1 = pt_backup
            cX = (M["m10"] / M["m00"])
            cY = (M["m01"] / M["m00"])
            mid_point.append(np.array([cX, cY]))
            
            pt_tmp = pt_next*ring_mask(frame, mid_point[-1], maxm/2*1.4, 3).astype(bool)
            pt_backup = pt_next*outer_mask(frame, mid_point[-1], maxm/2*1.4).astype(bool)
            pt_next = pt_next*outer_mask(frame, mid_point[-1], maxm/2*1.8).astype(bool)
            
            M = cv2.moments(pt_tmp)
        M = cv2.moments(pt_backup1)
        if M["m00"] != 0 :
            cX = (M["m10"] / M["m00"])
            cY = (M["m01"] / M["m00"])
            mid_point.append(np.array([cX, cY]))
            # xtmp = np.squeeze(cv2.findNonZero(pt_backup1))
            # xtmp1 = xtmp - np.array([cX, cY])
            # score = np.dot(xtmp1, mid_point[-1]-mid_point[-2])
            # mid_point.append(xtmp[score.argmax(),:])
                
        img, pos = self.update(idx, img0, pt, 3, inter, ibox, centerline=mid_point)
        
        return img, pos

    def motion_track(self, idx, img0, inter, ibox):      
        """ Motion tracking 
            fgMask: greyscale image showing the output of the background subtraction in the roi
        """
        img = cv2.medianBlur(img0,5) 
        frame = img[self.roi[2]:self.roi[3],self.roi[0]:self.roi[1]] 
        # For first frames, only update bg tracker
        if idx <= self.st + 4*self.sd:
            self.el_track = True
            bgfull = self.bg_tracker.apply(img)
            self.last_frame = img0.copy()
            self.last_bg_frame['id'] = idx
            self.last_bg_frame['frame'] = img.copy()
            self.last_bg_frame['roi'] = self.roi.copy()
            img, pos = self.update(idx, img0, frame, 2, inter, ibox)
            return img, pos
                
        # openCV optical flow
        flow = cv2.calcOpticalFlowFarneback(self.last_bg_frame['frame'], img, None, 0.5, 3, 25, 3, 7, 1.5, 0)
        # transform to polar coord
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv = np.zeros(( np.array(img).shape[0], np.array(img).shape[1], 3 ))
        # hsv[..., 1] = 255
        # hsv[..., 0] = ang*180/np.pi/2
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # hsv = hsv.astype(np.uint8) 
        # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # bgrt = bgr[self.roi[2]:self.roi[3],self.roi[0]:self.roi[1]]
        # cv2.imshow('frame2', bgr)
        
        # tmp is a mask to exclude previous PT region
        tmp = ~self.mot_traj[self.roi[2]:self.roi[3],self.roi[0]:self.roi[1]]
        
        # mot is the motion magnitude in the box (excluding prev PT)
        mot = mag[self.roi[2]:self.roi[3],self.roi[0]:self.roi[1]]*tmp
        # ang is the motion angle in the box (excluding prev PT)
        ang = ang[self.roi[2]:self.roi[3],self.roi[0]:self.roi[1]]*tmp 
             
        mask = circular_mask(frame, self.gamma).astype(bool)
        mask_flow = mot*mask
        
        # flow_mag = mag[self.roi[2]:self.roi[3],self.roi[0]:self.roi[1]]
        if len(mask_flow[mask_flow>0.2])>0:
            flow_mag_mean = mask_flow[mask_flow>0.2].mean()
        else:
            flow_mag_mean=0
        if idx%5>0:
            flow_mag_mean = 0
        if flow_mag_mean<0.2 and idx-self.last_bg_frame['id']<20: # use ellipse tracker
            self.el_track = True
            self.last_frame = img0.copy()
            # print("Optical flow failure: {:.2f}".format(flow_mag_mean))
            img, pos=self.update(idx, img0, frame, 2, inter, ibox, flow_mean=flow_mag_mean )
            return img, pos

        else: # motion great -> update roi; could be tuned! <--------------------
            # flow > threshold, apply bg tracker
            bgfull = self.bg_tracker.apply(img)
            bgfull = cv2.medianBlur(bgfull,3) # cv2.erode(fgMask, np.ones((3, 3)), 1)
            fgMask = bgfull[self.roi[2]:self.roi[3],self.roi[0]:self.roi[1]]  
            # _, fgMask = cv2.threshold(fgMask,180,255,cv2.THRESH_BINARY)
            
            self.last_bg_frame['id'] = idx
            self.last_bg_frame['frame'] = img.copy()
            self.last_bg_frame['roi'] = self.roi.copy() 

            ang = np.average(ang, weights = mot) # vital to use weight here
            mot_mag = np.average(mot, ) # not vital to use fgMask as weights here
            mask_fgMask = fgMask *tmp*mask
            cv2.imshow("111",fgMask)
            # cX, cY = int((self.roi[1]-self.roi[0])/2), int((self.roi[3]-self.roi[2])/2)
            # find contours
            contours0, _ = cv2.findContours(mask_fgMask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
            # self.find_PTs(frame, contours)
            self.el_track = True
            if len(contours0) != 0:
                # find the biggest countour (c) by the area
                tip = self.find_bg_tip(mask_fgMask)
                contours, _ = cv2.findContours(tip, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
                c = max(contours, key = cv2.contourArea)
                if cv2.contourArea(c) > 2.5:
                    # set up prev PT region
                    fill2 = np.zeros_like(fgMask) 
                    hull2 = cv2.convexHull(c)
                    cv2.fillPoly(fill2, pts =[hull2], color=255)
                    self.mot_traj[self.roi[2]:self.roi[3],self.roi[0]:self.roi[1]] |= fill2.astype(bool)
                    M = cv2.moments(c)
                    if M["m00"] != 0 : #and M["nu02"]+M["nu20"]<2
                        cX0 = (M["m10"] / M["m00"])
                        cY0 = (M["m01"] / M["m00"])
                        cX0 = self.roi[0] + cX0
                        cY0 = self.roi[2] + cY0
                        tar = np.array([cX0,cY0])
                        g = tar-self.center
                        movement = np.linalg.norm(g)
                        if len(self.center_list)>3:
                            dist_tmp = np.linalg.norm(self.center_list - tar, axis=1)
                            if 0 < movement < dist_tmp[:-3].min()+3:
                                self.el_track = False
                        # elif np.dot(g,self.gamma)>0:
                        #         self.el_track = False
                        elif movement<15:
                            self.el_track = False
            
            self.last_frame = img0.copy()
            if self.el_track:
                img, pos = self.update(idx, img0, frame, 2, inter, ibox, flow_mean=flow_mag_mean)

            else:        
                gamma =  g/np.linalg.norm(g)
                self.roi = update_roi(self.roi, g)
                if movement>20:
                    self.center_list = []
                img, pos = self.update(idx, img0, frame, 1, inter, ibox, gamma)
            # img, pos = self.update(idx, img0, frame, 1, inter, ibox, (cX,cY))

        return img, pos
  
    
    def update(self, idx, frame, temp, code, inter=False, ibox=None, gam=None,
               centerline=None, process=True, flow_mean=None):
        """
        Update center, roi, etc.
        Args:
            code (int):  0: using ellipse tracking, and update roi 
                         1: using motion tracking, and update center/roi 
                         2: using motion tracking, and NOT update roi 
                         3: using fluo tracking, and update roi 
        """
        
        self.last_frame = frame.copy()
        if process:
            # frame = img_process(frame)
            temp = img_process(temp)
        mask1 = circular_mask(temp, self.gamma)
        # hull, ellipse = thres_transform(temp, p1, (30,80))
        hull, ellipse = self.contour_transform(temp, [30,80], mask1, 0.33)
        sigma = 0.33
        while hull is None:
            hull, ellipse = self.contour_transform(temp, [30,80], mask1, sigma)
            sigma = sigma*1.2
            if sigma>0.85:
                break
        if hull is not None:
            roi_tmp, gamma_tmp, mag = update_bbox(ellipse, self.center, None, 
                                                  gamma=self.gamma, roi=self.roi.copy())
            
        if code == 1: # motion detection ok -> update
            tracker_active = "BG Tracker"
            # here cX, cY are still floats (for precise calculation of gamma)
            if hull is not None:
                gam = gam + gamma_tmp
            self.gamma =  gam/np.linalg.norm(gam)
            self.center = np.array([(self.roi[0]+self.roi[1])//2,(self.roi[2]+self.roi[3])//2])
            self.center_list.append(self.center.copy())
            cv2.circle(frame, tuple(self.center), 5, (255, 255, 255), -1)
            
        elif code == 2: # first 5 imgs for mot OR motion failed
            tracker_active = "Ellipse Tracker"
            if hull is not None:
                if flow_mean is not None and mag > flow_mean*5: # fasle detection
                    ellipse = self.bp_ellip
                else:
                    # self.roi = roi_tmp.copy()
                    self.gamma = gamma_tmp.copy()
                    self.bp_ellip = ellipse
            else:
                ellipse = self.bp_ellip
            self.center = np.array([(self.roi[0]+self.roi[1])//2,(self.roi[2]+self.roi[3])//2])
            
        elif code == 3: # fluo tracking update
            tracker_active = "Fluo Tracker"
            if hull is not None:
                tar = np.array([(roi_tmp[0]+roi_tmp[1])//2,(roi_tmp[2]+roi_tmp[3])//2])
                # TODO: here implement outlier detection
                
                self.roi = roi_tmp.copy()
                self.gamma = gamma_tmp.copy()
                self.center_list.append(tar.copy())
            
            pt0=np.array([self.roi[0],self.roi[2]])
            for i in range(len(centerline)-1):
                cv2.line(frame, tuple(centerline[i].astype(np.uint8)+pt0), 
                         tuple(centerline[i+1].astype(np.uint8)+pt0), 0, 2)
            try:
                g1 = centerline[-1]-centerline[-3]
                g2 = centerline[-2]-centerline[-3]
                g_ave = g1+g2
                self.gamma = g_ave/np.linalg.norm(g_ave)
            except:
                pass
            self.bp_ellip = ellipse
            self.center = np.array([(self.roi[0]+self.roi[1])//2,(self.roi[2]+self.roi[3])//2])
            centerline_global = list((centerline + pt0).astype(int))
        
        else: # using ellipse tracking (0)
            tracker_active = "Ellipse Tracker"
            if hull is not None:
                tar = np.array([(roi_tmp[0]+roi_tmp[1])//2,(roi_tmp[2]+roi_tmp[3])//2])
                if len(self.center_list)>3:
                    dist_tmp = np.linalg.norm(self.center_list - tar, axis=1)
                    if 0 < mag < dist_tmp[:-3].min()+3:
                        self.roi = roi_tmp.copy()
                        self.gamma = gamma_tmp.copy()
                        self.center_list.append(tar.copy())
                        cv2.ellipse(frame, ellipse, (255,0,0), 2)
                elif idx > self.st + 2:
                    pre_idx = self.idx_list[-2]
                    if np.dot(gamma_tmp,self.data[pre_idx]['gamma'])>0 and np.dot(
                        tar-self.data[pre_idx]['center'],self.data[pre_idx]['gamma'])>0:
                        self.roi = roi_tmp.copy()
                        self.gamma = gamma_tmp.copy()
                        self.center_list.append(tar.copy())
                        cv2.ellipse(frame, ellipse, (255,0,0), 2)
                else:
                    self.roi = roi_tmp.copy()
                    self.gamma = gamma_tmp.copy()
                    self.center_list.append(tar.copy())
                    cv2.ellipse(frame, ellipse, (255,0,0), 2)
               
                self.bp_ellip = ellipse
                # cv2.ellipse(frame, ellipse, (255,0,0), 2)
                # cX, cY = el2center(self.bp_ellip, self.gamma)
                # cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
              
            elif self.bp_ellip is not None:
                ellipse = self.bp_ellip
            self.center = np.array([(self.roi[0]+self.roi[1])//2,(self.roi[2]+self.roi[3])//2])
            
        p1, p2 = roi2pt(self.roi)
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        # cv2.arrowedLine(frame,tuple(self.center),tuple((self.center+self.gamma*30).astype(np.uint16)),(255, 255, 255), 5)
        
        # score tracking data
        self.data[idx] = {'center': self.center.copy(), 'ellipse': ellipse,
                          'roi': self.roi, 'gamma': None, 'frame': self.last_frame.copy() }  
        if self.gamma is not None:
            self.data[idx]['gamma'] = self.gamma.copy() 
        if centerline is not None:
            self.data[idx]['centerline'] = centerline.copy() 
            self.data[idx]['centerline_global'] = centerline_global.copy()
        
        """ --------------- track interface ------------ """
        if inter:
            try:
                if self.bp_lines is not None:
                    # dist_l2c = distance(self.bp_lines, self.center)
                    # min_dist = dist_l2c.min(axis=0)
                    # min_index = dist_l2c.argmin()
                    # in debug mode: >=0; in practice: ==0
                    if True: # (idx//self.sd)% 2 == 0: # min_dist < min(temp.shape)/3 # TODO <=============================
                        lines = self.interface(ibox, p1, p2, code)
                        # lines = blend_lines(lines, self.bp_lines, 0.3, 0.7)
                        self.bp_lines = lines.copy()
                    else:
                        lines = self.bp_lines.copy()
                        
                else:
                    lines = self.interface(ibox, p1, p2, code)
                    self.bp_lines = lines.copy()
                
                if code == 3:
                    self.data[idx]['inter'] = lines.copy()
                    for intf1 in lines:
                        if intf1 is None:
                            continue
                        for j in range(intf1.shape[0]):
                            cv2.circle(frame, (intf1[j,0], intf1[j,1]), radius=2, 
                                       color=(255,255,255), thickness=-1)
                else:
                    dist_l2c = distance(lines, self.center)
                    min_dist = dist_l2c.min(axis=0)
                    loi = lines[dist_l2c.argmin(),:]
                    loi = np.expand_dims(loi, axis=0)
                    self.data[idx]['inter'] = loi.copy()
                    
                    # local detection around PT
                    if min_dist < 0: #min(temp.shape)//1.4:
                        local_inter = self.local_inter(p1, p2)
                        if 'local_inter' in self.data[self.idx_list[-2]].keys():
                            local_inter = blend_lines(local_inter, self.data[self.idx_list[-2]]['local_inter'], 0.4, 0.6)
                        self.data[idx]['local_inter'] = local_inter.copy()
        
                        drawhoughLinesOnImage(frame, loi, self.center, self.gamma, local_inter)
                    
                    else:
                        drawhoughLinesOnImage(frame, loi, self.center, self.gamma)
                
                # return PT position in the channel
                theta = self.walls['angle'][-1]
                rho = self.center[0]*np.cos(theta)+self.center[1]*np.sin(theta)
                w1 = self.walls['wall1'][-1]
                w2 = self.walls['wall2'][-1]
                pos = [(rho-w1)/(w2-w1), (w2-rho)/(w2-w1), w2-w1, theta]
            except:
                print("Interface Error.")
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                del exc_info
                pos = []
        else:
            pos =[]

        # Display tracker type on frame
        cv2.putText(frame, tracker_active, (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255,2); # (50,170,50)
    
        # Display frame No.
        cv2.putText(frame, "Frame : " + str(int(idx)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2);
                
        return frame, pos
    
    def contour_transform(self, gray, cann =[80,180], mask=None, sigma=0.33):
        v=np.median(gray)
        cann[0] = int(max(0, (1.0 - 2*sigma) * v))
        cann[1] = int(min(255, (1.0 - sigma) * v))
        edged=cv2.Canny(gray,cann[0],cann[1], L2gradient = True)
        # cv2.imshow( 'edged', edged)
        edged = self.find_bg_tip(edged, eps=15)
        # cv2.imshow( 'edged2', edged)
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        # hull = canny_t(gray, cann)
        
        if contours:          
            comb = np.vstack(contours)
            hull = cv2.convexHull(comb)
            fill1 = np.zeros_like(edged) 
            cv2.fillPoly(fill1, pts =[hull], color=255)
            
            if mask is not None:
                fill1 = fill1 & mask
                # cv2.imshow('masked final', cv2.pyrUp(fill1))
            
            fill1 = cv2.findNonZero(fill1)
            if fill1 is not None and fill1.shape[0] > 5:
                hull = cv2.convexHull(fill1)
                ellipse = list(cv2.fitEllipse(fill1))
            else:
                return None, None

            # process display in original image
            hull[:,:,0]+= self.roi[0]
            hull[:,:,1]+= self.roi[2]
            ellipse[0] = (ellipse[0][0]+self.roi[0], ellipse[0][1]+self.roi[2])
            ellipse = tuple(ellipse)
            return hull, ellipse
        else:
            return None, None
        
    def traj_update(self, ):
        pass

    def find_PTs(self, frame, contours, n=4, eps=20, msp=20):  # eps=30
        collec = np.zeros((1,2), dtype=int)
        for i in range(len(contours)):
            c = contours[i]
            collec=np.vstack((collec, np.squeeze(c)))
        collec=collec[1:,:]
        
        # perform DBSCAN clustering (more robust to outliers)
        db = DBSCAN(eps=eps, min_samples=msp)
        db.fit(collec)
        labels = db.labels_
        mask = np.zeros_like(labels, dtype=bool)
        mask[db.core_sample_indices_] = True
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        
        # find centers of each cluster
        centers = np.zeros((n_clusters_, collec.shape[1]+1))
        for k in range(n_clusters_):
            member_mask = labels == k
            member = collec[mask & member_mask]
            centers[k,:-1] = np.mean(member, axis=0)
            num = np.count_nonzero(member_mask)
            centers[k,-1] = -num
        
        centers = centers[centers[:,-1].argsort(), :]
        if n<= n_clusters_:
            centers = centers[:n,:-1]
        else:
            centers = centers[:,:-1]
            
        # # test clustering result
        # drawing = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        # colors=[(0,255,0),(0,0,255),(255,0,0),(50,50,50),(120,200,20),(20,100,200),
        #         (220,20,20)]
        # tmp=0
        # for i in range(len(contours)):
        #     c=contours[i]
        #     if labels[tmp]>=0:
        #         cv2.drawContours(drawing, [c], 0, colors[labels[tmp]])
        #     tmp+=c.shape[0]
        # cv2.imshow('Contours',  drawing)
        
        y=dbscan_predict(db, self.center)
        member_mask = labels == y
        member = collec[mask & member_mask]
        tmp_=member.copy()
        tmp_[:,0]=tmp_[:,1]
        tmp_[:,1]=member[:,0]
        filled = np.zeros_like(frame) 
        filled[tuple(np.transpose(tmp_))]=255
        _, contours1, _ = cv2.findContours(filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # test contour
        drawing = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        comb = np.vstack(contours1)
        # approx = cv2.approxPolyDP(comb, 10, True)
        cv2.drawContours(drawing, [comb], 0, (255,0,0))
        cv2.imshow('Contours1',  drawing)
            
        return contours1