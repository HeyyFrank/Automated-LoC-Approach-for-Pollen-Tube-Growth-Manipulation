# -*- coding: utf-8 -*-
"""
MOSSE Tracker.
"""
import numpy as np
from sklearn.cluster import KMeans,DBSCAN
import math
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys
import time
import copy
import random as rng
from utils import *
from scipy.signal import find_peaks


class mo_Tracker:
    def __init__(self, tracker_type='MOSSE', ):
        major_ver, minor_ver, subminor_ver = (cv2.__version__).split('.')
        if int(minor_ver) < 3:
            self.tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                self.tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                self.tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                self.tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                self.tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                self.tracker = cv2.TrackerGOTURN_create()
            if tracker_type == 'MOSSE':
                self.tracker = cv2.TrackerMOSSE_create()
            if tracker_type == "CSRT":
                self.tracker = cv2.TrackerCSRT_create()
        self.tracker_type = tracker_type
        self.data = dict()
        
    def init_tracker(self, idx, frame, bbox, process=True):
        self.cur_idx = idx
        self.last_frame = frame.copy()
        if process:
            frame = img_process(frame)

        # select a bounding box
        # bbox = cv2.selectROI(frame, True)
        
        roi = np.array(bbox).reshape((2,2))
        roi[1,:] = roi[1,:]+roi[0,:]
        # cv2.destroyAllWindows()
        
        # Extract template and init temp tracker
        self.template = copy.deepcopy(frame[roi[0,1]:roi[1,1],roi[0,0]:roi[1,0]])

        # Initialize tracker with first frame and bounding box
        ok = self.tracker.init(frame, bbox)
        
        self.bp_bbox = bbox
        self.bp_ellip = None
        self.bp_lines = None
        self.center = np.array([bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2])
        self.gamma = None
        self.queue = []
        
        # self.last_frame = frame.copy()
        self.data[idx] = {'center': self.center.copy(), 'ellipse': None,
                          'bbox': bbox, 'gamma': None, 'frame': self.last_frame.copy()}

        return ok
    
    def update(self, idx, frame, inter=False, ibox=None, process=True, ):
        self.cur_idx = idx
        self.last_frame = frame.copy()
        if process:
            frame = img_process(frame)
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = self.tracker.update(frame)
        # ok=0

        # Draw bounding box
        if ok:
            # Tracking success            
            bbox = blend_box(bbox,self.bp_bbox, 0.01, 0.99)
            
        else :
            # Tracking failure
            ok = 1
            bbox = self.bp_bbox
            p1, p2 = None, None
            cv2.putText(frame, "Tracking failure detected", (100,880), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        
        p1, p2 = bbox2pt(bbox)
        self.center = np.array([bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2])
        temp = frame[p1[1]:p2[1],p1[0]:p2[0]].copy()
        mask1 = circular_mask(temp,self.gamma)
        
        # hull, ellipse = thres_transform(temp, p1, (30,80))
        hull, ellipse = self.contour_transform(temp, p1, p2, (30,80), mask1)
        if hull is not None and self.bp_ellip is not None:
            # ok = check_vad(ellipse,self.bp_ellip)
            ellipse, ok = blend_ellip(ellipse, self.bp_ellip, 0.95, 0.05)

        if hull is not None:
            
            # cv2.drawContours(frame, [hull], 0, (255, 0, 0), thickness = 2)
            # box = cv2.boxPoints(ellipse)
            self.bp_bbox, self.gamma = update_bbox(ellipse, self.center, bbox, self.gamma)
            
            self.center = np.array([self.bp_bbox[0]+self.bp_bbox[2]//2,self.bp_bbox[1]+self.bp_bbox[3]//2])
            ellipse = inert_ellip(ellipse, self.gamma, bbox[2]/20)
            # in case of false detection, retain previous ellepse
            if idx>1:
                # sudden change of growing direction > 45 DEG indicates false detection
                if np.dot(self.gamma,self.data[idx-1]['gamma'])<0.707 or np.dot(
                        self.center-self.data[idx-1]['center'],self.data[idx-1]['gamma'])<0.707:
                    # print("False detection!")
                    self.gamma = self.data[idx-1]['gamma'].copy()
                    ellipse = self.bp_ellip
                    self.bp_bbox = self.data[idx-1]['bbox']
                    self.center=self.data[idx-1]['center'].copy()
            
            p1, p2 = bbox2pt(self.bp_bbox)
            cv2.ellipse(frame, ellipse, (255,0,0), 2)
            self.bp_ellip = ellipse
        elif self.bp_ellip is not None:
            cv2.ellipse(frame, self.bp_ellip, (255,0,0), 2)
            ellipse = self.bp_ellip
        
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        
        # score tracking data
        self.data[idx] = {'center': self.center.copy(), 'ellipse': ellipse,
                          'bbox': self.bp_bbox, 'gamma': self.gamma.copy(), 'frame': self.last_frame.copy() }        
        # track interface
        # h, w = frame.shape
        # ibox = [w//2, 0, w-w//2-1, h-1]
        if inter:
            if self.bp_lines is not None:
                dist_l2c = distance(self.bp_lines, self.center)
                min_dist = dist_l2c.min(axis=0)
                min_index = dist_l2c.argmin()
                if min_dist < min(temp.shape)/3 and idx% 20 > 0: # TO MODIFY <=============================
                    lines = self.bp_lines.copy()
                else:
                    lines = self.interface(self.last_frame, ibox, p1, p2)
                    lines = blend_lines(lines, self.bp_lines, 0.3, 0.7)
                    self.bp_lines = lines.copy()
            else:
                lines = self.interface(self.last_frame, ibox, p1, p2)
                self.bp_lines = lines.copy()
                
            dist_l2c = distance(lines, self.center)
            min_dist = dist_l2c.min(axis=0)
            loi = lines[dist_l2c.argmin(),:]
            loi = np.expand_dims(loi, axis=0)
            self.data[idx]['inter'] = loi.copy()
            
            # local detection around PT
            if min_dist < min(temp.shape)//1.4:
                local_inter = self.local_inter(p1, p2)
                if 'local_inter' in self.data[self.cur_idx-1].keys():
                    local_inter = blend_lines(local_inter, self.data[self.cur_idx-1]['local_inter'], 0.4, 0.6)
                self.data[idx]['local_inter'] = local_inter.copy()

                drawhoughLinesOnImage(frame, loi, self.center, self.gamma, local_inter)
            
            else:
                drawhoughLinesOnImage(frame, loi, )

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # Display tracker type on frame
        cv2.putText(frame, self.tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255,2); # (50,170,50)
    
        # Display FPS on frame
        cv2.putText(frame, "Frame : " + str(int(idx)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2);
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2);
        
        # Display result
        # cv2.imshow("Tracking", frame)
        # cv2.waitKey(500)
        
        return ok, frame, p1, p2
    
    def contour_transform(self, gray, p1, p2, cann=(80,180), mask=None):
        fill1 = canny_t(gray, cann)
        if fill1 is not None:
            fill2 = self.motion_transform(p1, p2, cann=(20,50))
            if fill2 is not None:
                fill1 = fill1 & fill2   # | -> &
                # fill1 = fill2
            if mask is not None:
                fill1 = fill1 & mask
                # cv2.imshow('masked final', cv2.pyrUp(fill1))
            fill1 = cv2.findNonZero(fill1)
            hull = cv2.convexHull(fill1)
            ellipse = list(cv2.fitEllipse(fill1))
            
            # test contour
            # drawing = np.zeros((fill1.shape[0], fill1.shape[1], 3), dtype=np.uint8)
            # for i in range(1):
            #     # color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            #     # cv2.drawContours(drawing, [comb], i, color)
            #     cv2.drawContours(drawing, [hull], i, 255)
            # # cv2.imshow('Contours', drawing)
        
            # process display in original image
            hull[:,:,0]+= p1[0]
            hull[:,:,1]+= p1[1]
            ellipse[0] = (ellipse[0][0]+p1[0], ellipse[0][1]+p1[1])
            ellipse = tuple(ellipse)
            return hull, ellipse
        else:
            return None, None

    def motion_transform(self, p1, p2, cann=(20,50)):
        cur_frame = gaussian(self.last_frame, 5)
        if self.cur_idx-11 in self.data:
            pre_frame = gaussian(self.data[self.cur_idx-11]['frame'], 5)
            dif_th = 9
        else:
            return None
            pre_frame = gaussian(self.data[0]['frame'], 5)
            dif_th = 5
        # calculate difference and update previous frame
        diff_frame = cv2.absdiff(src1=pre_frame, src2=cur_frame)
        diff_frame = median(diff_frame,5)

        # Dilute the image a bit to make differences more seeable; more suitable for contour detection
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)
        
        # Only take different areas that are different enough (>20 / 255)
        thresh_frame = cv2.threshold(src=diff_frame, thresh=dif_th, maxval=255, type=cv2.THRESH_BINARY)[1]
        
        temp = thresh_frame[p1[1]:p2[1],p1[0]:p2[0]]
        # cv2.imshow( 'threshold', cv2.pyrUp(temp))
        hull = cv2.findNonZero(temp)
        hull = cv2.convexHull(hull)
        filled = np.zeros_like(temp) 
        cv2.fillPoly(filled, pts =[hull], color=255)
        filled = cv2.dilate(filled, kernel, 2)
        
        
        # second canny edge detection
        temp2 = cur_frame[p1[1]:p2[1],p1[0]:p2[0]]
        edge = cv2.Canny(temp2,cann[0],cann[1], L2gradient = True)
        # cv2.imshow( 'edged', cv2.pyrUp(filled_),)

        filled = filled & edge
        # cv2.imshow( 'threshold', cv2.pyrUp(filled))
        
        # # test contour
        # _, contours, _ = cv2.findContours(filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #  # test contour
        # drawing = np.zeros((filled.shape[0], filled.shape[1], 3), dtype=np.uint8)
        # for i in range(len(contours)):
        #     color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        #     # cv2.drawContours(drawing, [comb], i, color)
        #     cv2.drawContours(drawing, [contours[i]], 0, color)
        # cv2.imshow('Contours',  cv2.pyrUp(drawing))
        
        return filled
        
    def interface(self, frame, ibox, pt1, pt2):
        p1, p2 = bbox2pt(ibox)
        temp = frame[p1[1]:p2[1],p1[0]:p2[0]]
        
        # process image in the box and get two boundary lines
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
        dst = clahe.apply(frame)
        dst = img_process(dst)
        temp = dst[p1[1]:p2[1],p1[0]:p2[0]]
        edged = cv2.Canny(temp, 50, 180, L2gradient = True) # obv: 50, 120
        # cv2.imshow('canny edges', edged)
        houghLines = self.hough(edged, n=2)
        houghLines = houghLines[houghLines[:,0].argsort(), :]
        houghLines = coord_trans(houghLines, p1)

        # find the interfaces parallel to boundary lines
        th3 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,35,-1)
        # cv2.imshow( 'threshold', th3,)
        result = self.search_int(th3, houghLines, p1[0], p2[0])
        theta = np.average(houghLines[:,1])
        houghLines = np.array([[result[0], theta],[result[1],theta]])
        
        # # local detection around PT
        # self.local_inter(pt1, pt2)
        
        return houghLines
        
    def hough(self, edged, n=4, th=50, eps=50, ms=5):
        """ Hough Line detect """
        dis_reso = 1 # Distance resolution in pixels of the Hough grid
        theta = np.pi /180 # Angular resolution in radians of the Hough grid
        threshold = th # minimum no of votes 100
        
        houghLines = cv2.HoughLines(edged, dis_reso, theta, threshold)
        if houghLines is None:
            return None
        lines = np.squeeze(houghLines)
        if len(lines.shape)<2:
            lines = np.expand_dims(lines, axis=0)
            
        # perform DBSCAN clustering (more robust to outliers)
        db = DBSCAN(eps=eps, min_samples=ms)
        db.fit(lines)
        labels = db.labels_
        mask = np.zeros_like(labels, dtype=bool)
        mask[db.core_sample_indices_] = True
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        centers = np.zeros((n_clusters_, lines.shape[1]+1))
        for k in range(n_clusters_):
            member_mask = labels == k
            member = lines[mask & member_mask]
            centers[k,:-1] = np.mean(member, axis=0)
            num = np.count_nonzero(member_mask)
            centers[k,-1] = -num
        
        centers = centers[centers[:,-1].argsort(), :]
        if n<= n_clusters_:
            centers = centers[:n,:-1]
        else:
            centers = centers[:,:-1]
            
        # perform Kmeans clustering
        # kmeans = KMeans(n_clusters=n, random_state=0)
        # kmeans.fit(lines)
        # centers = kmeans.cluster_centers_
        
        return centers

    def local_inter(self, p10, p20, cann=(50,100), lamda=1.5):
        # have a larger area than PT roi to search for interface
        p10=np.array(p10)
        p20=np.array(p20)
        p2=(p10+lamda*(p20-p10)).copy().astype(int)
        p1=p10.copy()
        x0=np.array([1,0])
        y0=np.array([0,1])
        if self.gamma @ x0<0:
            p1[0]-=p2[0]-p20[0]
            p2[0]-=p2[0]-p20[0]
        if self.gamma @ y0<0:
            p1[1]-=p2[1]-p20[1]
            p2[1]-=p2[1]-p20[1]
        
        # down sample the area and process img
        frame = img_process(self.last_frame)
        temp = frame[p1[1]:p2[1],p1[0]:p2[0]].copy()
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
        dst = clahe.apply(temp)
        dst = gaussian(dst, 5)
        dst = cv2.pyrDown(dst)
        edge2 = cv2.Canny(dst, cann[0], cann[1], L2gradient = True)
        # cv2.imshow('canny edges 2', cv2.pyrUp(edge2))
        
        # construct a mask to filter out PT edges
        mask = np.ones_like(frame)*255
        cv2.ellipse(mask, self.bp_ellip, 0, -1)
        kernel = np.ones((7, 7))
        mask = cv2.erode(mask, kernel, 15)
        mask = mask[p1[1]:p2[1],p1[0]:p2[0]]
        mask = cv2.pyrDown(mask)
        
        edge2 = edge2 & mask
        # cv2.imshow('masked', cv2.pyrUp(edge2))
        # hough transform to find the most possible interface
        ok=1
        th=28
        while ok:
            lines = self.hough(edge2, th=th, eps=10, ms=1, n=1)
            if lines is None:
                th-=2
                print(th+2, 'failed')
            else:
                ok = 0
        # median filter
        self.queue.append(lines[0])
        # if len(self.queue)>15:
        #     self.queue.pop(0)
        q = np.array(self.queue)[-11:,:]
        med = np.median(q, axis=0)
        weight = 1/(np.linalg.norm(q-med,axis=1)+0.02)
        weight = np.expand_dims(weight/np.sum(weight),1)
        final_line = np.sum(q*weight, axis=0)
        lines = np.expand_dims(final_line, axis=0)
        
        # show
        houghImage = np.zeros_like(dst) # create and empty image
        drawhoughLinesOnImage(houghImage, lines)
        imgFinal = blend_images(houghImage, dst, alpha=.5,)
        # cv2.imshow('final', cv2.pyrUp(imgFinal))
        
        # transform to ori coord
        lines[:,0]*=2
        lines = coord_trans(lines, p1)
        # check outlier
        if 'local_inter' in self.data[self.cur_idx-1].keys():
            # linep = self.data[self.cur_idx-1]['local_inter']
            linep = np.median(np.array(self.queue)[-21:,:], axis=0)
            linep = np.expand_dims(linep, axis=0)
            if abs((linep-lines)[0,1])>np.pi/4:
                lines = linep.copy()
        return lines
    
    def search_int(self, img, lines, st=50, ed=600):
        theta = np.average(lines[:,1])
        # m = np.cos(theta)/np.sin(theta)
        p1 = int(np.min(lines[:,0]))
        p2 = int(np.max(lines[:,0]))
        dist = (p2-p1)//16
        dist0 = (p2-p1)//8
        p1 = p1 + dist
        p2 = p2 - dist
        # if self.bp_lines is not None:
        #     p1 = max(p1, int(min(self.bp_lines[:,0])-dist0))
        #     p2 = min(p2, int(max(self.bp_lines[:,0])+dist0))
        cum = np.zeros(p2-p1)
            
        # Search general lines
        for dr in range(p1,p2):
            for col in range(st,ed):
                cum[dr-p1] += img[round((dr-col*np.cos(theta))/np.sin(theta)),col]
            cum[dr-p1] = cum[dr-p1]/(ed-st)
        
        avg = np.average(cum)
        peaks, _ = find_peaks(cum,  distance=30, height=avg)
        peaks = np.stack((peaks,cum[peaks]))
        peaks = peaks[:, peaks[1, :].argsort()]
        
        valleys, _ = find_peaks(cum*(-1),  distance=30, height=-avg)
        valleys = np.stack((valleys,cum[valleys]))
        valleys = valleys[:, valleys[1, :].argsort()]
        result=[]
        for i in range(2):
            candpk = int(peaks[0,-(i+1)])
            candvl = int(valleys[0, i ])
            if abs(candpk-candvl)<40:
                result.append((candpk+candvl)//2+p1)
                if len(result)>=2:
                    break
            else:
                if avg-valleys[1,i]> peaks[1, -i-1]:
                    result.append(candvl+p1)
                    # result.append(candpk+p1)
                else:
                    # result.append(candpk+p1)
                    result.append(candvl+p1)
                if len(result)>=2:
                    break

        return result
        