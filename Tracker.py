# -*- coding: utf-8 -*-
"""
Abstract Tracker class.
"""
import numpy as np
from sklearn.cluster import KMeans,DBSCAN, SpectralClustering
from sklearn.mixture import BayesianGaussianMixture
import math
import cv2
import sys
import time
import copy
import random as rng
from utils import *
from scipy.signal import find_peaks, convolve2d
from scipy.stats import iqr
from scipy.optimize import curve_fit
from statistics import median
import traceback

class Tracker:
    def __init__(self ):
        self.tracker_type = tracker_type
        self.data = dict()
        
    def check_outlier(self, lines):
        if len(self.walls['wall1'])<2 or len(self.walls['wall2'])<2:
            self.walls['angle'].append(lines[0][1])
            self.walls['wall1'].append(lines[0][0])
            self.walls['wall2'].append(lines[1][0])
            return lines
        theta = median(self.walls['angle'])
        iqr_th = iqr(self.walls['angle'])
        rho=[]
        iqr_r=[]
        rho.append(median(self.walls['wall1']))
        iqr_r.append(iqr(self.walls['wall1']))
        rho.append(median(self.walls['wall2']))
        iqr_r.append(iqr(self.walls['wall2']))
        for i in range(2):
            if rho[i]-0.5*iqr_r[i]<=lines[i][0]<=rho[i]+0.5*iqr_r[i] and \
                theta-iqr_th*0.5<=lines[i][1]<=theta+iqr_th*0.5:
                self.walls['angle'].append(lines[i][1])
                if len(self.walls['angle'])>self.walls['len']:
                    self.walls['angle'].pop(0)
                self.walls['wall'+str(i+1)].append(lines[i][0])
                if len(self.walls['wall'+str(i+1)])>self.walls['len']:
                    self.walls['wall'+str(i+1)].pop(0)
        return np.array([[self.walls['wall1'][-1],self.walls['angle'][-1]],\
                [self.walls['wall2'][-1],self.walls['angle'][-1]]])
    
    def interface(self, ibox, pt1, pt2, code):
        """
        Variables:
            roi: 4x small frame;
            th0: thresholded 4x small frame;
            int_code [0/1]: defining the direction of interface detection with regard to PT
                            0-growing to right; 1-growing to left
        """
        p1, p2 = bbox2pt(ibox)
        frame = self.last_frame        
        # Using fluo
        if code == 3:
            thv = np.median(frame)+np.std(frame)*3.0
            if thv<120:
                ret,img = cv2.threshold(frame,thv,255,cv2.THRESH_TRUNC ) 
                img = (img/img.max())
                img = (img*255).astype(np.uint8)
            else:
                img = frame

            # img pyramid: 4x smaller img for line detection
            roi = img.copy()
            roi= cv2.pyrDown(roi)
            roi = cv2.pyrDown(roi)
            roi = img_process(roi, 3)
            
            try: # auto adjust denoising/bluring based on SNR & num of interface points
                if self.SNR>=self.SNR_level[0]:
                    pass
                elif self.SNR<self.SNR_level[0] and self.SNR>self.SNR_level[1]:
                    self.n_denoise = min(3, self.n_denoise+1)
                elif self.SNR<=self.SNR_level[1]:
                    self.n_denoise = 3
                if self.n_SNR_data < self.SNR_level[2]:
                    self.n_denoise = max(self.n_denoise-1, 1)
                for i in range(self.n_denoise):
                    roi = img_process(roi, 3+2*(i%2))
                # print("{:d}, {:.2f}, {:d}".format(self.n_denoise, self.SNR, self.n_SNR_data))
            except:
                self.n_denoise = 1
                roi = img_process(roi, 3)
            
            intf_th=25
            th0 = cv2.adaptiveThreshold(roi, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,intf_th,-1)
            # select loi we selected for boundry line detection
            th1 = th0[p1[1]//4:p2[1]//4,p1[0]//4:p2[0]//4]
            # cv2.imshow( 'threshold 1', th0,)
            lines = self.hough(th1, n=4, eps=5)
            lines = lines[lines[:,0].argsort(), :]
            # process the lines to get both boundary and interface samples
            # n_clusters = lines.shape[0]-2
            # print(n_clusters)
            # if n_clusters < 1:
            #     return None
            # elif n_clusters == 1:
            #     cand_int = np.array([lines[1,:]])
            # else:
            #     cand_int = np.array([lines[1,:],lines[-2,:]])
            
            # select boundary lines
            lines = np.array([lines[0,:],lines[-1,:]])
            boundLines = lines.copy()
            boundLines[:,0]*=4
            boundLines = coord_trans(boundLines, p1)
            try:
                boundLines = self.check_outlier(boundLines)
            except:
                pass
            lines = coord_trans(lines, np.array(p1)/4)
            try:
                fluoLines = self.fluoBoundLines
                if lines[0,0] < fluoLines[0,0] and abs(lines[0,1]-fluoLines[0,1])<0.04:
                    fluoLines[0,:] = lines[0,:]
                if lines[1,0] > fluoLines[1,0] and abs(lines[1,1]-fluoLines[1,1])<0.04:
                    fluoLines[1,:] = lines[1,:]
                self.fluoBoundLines = fluoLines.copy()
            except:
                self.fluoBoundLines = lines.copy()
                fluoLines = lines

            # extract interface region
            mask1 = np.zeros_like(roi)
            mask2 = np.zeros_like(roi)
            mask1[:5,:]=1
            mask2[-5:,:]=1
            Xs = []
            for m in [mask1, mask2]:
                tmp = th0*m.astype(bool)
                M = cv2.moments(tmp)
                if M["m00"] != 0 :
                    cX = (M["m10"] / M["m00"])
                    cY = (M["m01"] / M["m00"])
                    Xs.append(cX)
            try:
                t1, t2, t3, t4 = self.loi2[0], self.loi2[1], self.loi2[2], self.loi2[3]
                int_code = self.int_code
                if int_code==0:
                    t1 = max(t1, (self.center[0]-(self.roi[1]-self.roi[0])//2)//4)
                    t2 = p2[0]//4
                else:
                    t2 = min(t2, (self.center[0]+(self.roi[1]-self.roi[0])//2)//4)
                    t1 = p1[0]//4
            except:
                if self.center[0]<p1[0]:
                    t1 = (self.center[0]-(self.roi[1]-self.roi[0])//2)//4
                    t2 = p2[0]//4
                    self.int_code = 0
                else:
                    t1 = p1[0]//4
                    t2 = (self.center[0]+(self.roi[1]-self.roi[0])//2)//4
                    self.int_code = 1
                    
            tmp=[]
            for line in fluoLines:
                rho,theta = line[0],line[1]
                a = np.cos(theta)
                b = np.sin(theta)
                y1 = -t1*a/b+rho/b
                y2 = -t2*a/b+rho/b
                tmp.append(y1)
                tmp.append(y2)
            t3 = int(max(tmp[0:2]))
            t4 = int(min(tmp[2:4]))
                
            pt_mask = cv2.dilate(cv2.pyrDown(cv2.pyrDown(self.fluo_traj.astype(
                np.uint8))),np.ones((5,5),np.uint8),iterations = 1)
            # th2 = th0 * ~(pt_mask).astype(bool)     
            # th2 = th2[t3:t4,t1:t2]
            th22 = roi[t3:t4,t1:t2]
            th2 = cv2.adaptiveThreshold(th22, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,intf_th,-1)
            th2 = th2 * ~(pt_mask[t3:t4,t1:t2]).astype(bool)
            # cv2.imshow( 'threshold 1.5', th2,)
            try:
                self.loi2 = [t1, t2, t3, t4]
                # lines = self.hough(th2, n=2, eps=5)
                # lines = lines[lines[:,0].argsort(), :]
                # print(lines)
                self.far_intfs = self.sort_interface(th2[:,max(0,p1[0]//4-t1):], 
                                                     (p1[0], 4*t3), roi, th1) # th0[t3:t4,p1[0]//4:p2[0]//4]
                if self.intf_reg == 'fixed':
                    # only consider loi region
                    intfs = self.far_intfs.copy()
                else:
                    intfs = self.sort_interface(th2, (4*t1, 4*t3), roi, th1)
            except Exception:
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                del exc_info
                print("Error defining LOI: ")
                intfs = None
            return intfs
            
        else:
            # process image in the box and get two boundary lines
            clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
            dst = clahe.apply(frame)
            dst = img_process(dst)
            temp = dst[p1[1]:p2[1],p1[0]:p2[0]]
            edged = cv2.Canny(temp, 50, 180, L2gradient = True) # obv: 50, 120
            houghLines = self.hough(edged, n=2)
            houghLines = houghLines[houghLines[:,0].argsort(), :]
            houghLines = coord_trans(houghLines, p1)
            boundLines = self.check_outlier(houghLines)
            th3 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,35,-1)
            # roi = frame.copy()
            # roi= cv2.pyrDown(roi)
            # roi = cv2.pyrDown(roi)
            # roi = img_process(roi)
            # th0 = cv2.adaptiveThreshold(roi, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #     cv2.THRESH_BINARY,25,1)
            # cv2.imshow('frame1', th0)
            # find the interfaces parallel to boundary lines
            result = self.search_int(th3, boundLines, p1, p2)
            theta = np.average(boundLines[:,1])
            houghLines = np.array([[result[0], theta],[result[1],theta]])
            
            # # local detection around PT
            # self.local_inter(pt1, pt2)
            return houghLines
        
    def sort_interface(self, img, p1, roi, loi, eps=10, msp=30):
        """ 
        Cluster the thresholding mask with DBSCAN, select the 2 interfaces and 
        return the interfacesn with sampled points
        Args:
            intfs [intf1, intf2]: list containing the two interfaces, where
                each intf is a [200*2] array containing sampled interface points 
        """
        # def func(x, a, b, c):
        #     return a * np.exp(-b * x) + c
        
        # imgCut = img[img.shape[0]//2:, :]
        # c1 = cv2.findNonZero(imgCut)
        # c1 = np.squeeze(c1)

        # process roi for curved int
        collec = cv2.findNonZero(img)
        collec = np.squeeze(collec)
        
        """ GMM """
        # seed = np.random.choice(collec.shape[0], min(300,collec.shape[0]), replace = False)
        # X=collec[seed]
        # # collec=collec[seed]
        # bgm = BayesianGaussianMixture(n_components=4, random_state=0,max_iter=100).fit(X)
        # Y_=bgm.predict(X)
        # color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])
        # plt.close()
        # for i, color in enumerate( color_iter):

        #     if not np.any(Y_ == i):
        #         continue
        #     plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)
        # plt.show()
        
        """ Fit curve """
        # seed = np.random.choice(c1.shape[0], min(50,c1.shape[0]), replace = False)
        # X=c1[seed]
        # # collec=collec[seed]
        # x=X[:,0]
        # y=X[:,1]
        # popt, pcov = curve_fit(func,  x,  y)
        # plt.figure()
        # plt.plot(x, y, 'ko', label="Original Noised Data")
        # x2 = np.linspace(1,img.shape[1],50)
        # plt.plot(x2, func(x2, *popt), 'r-', label="Fitted Curve")
        # plt.legend()
        # plt.show()
        
        """ Simple DBSCAN """
        # perform DBSCAN clustering (more robust to outliers)
        db = DBSCAN(eps=eps, min_samples=msp)
        db.fit(collec)
        labels = db.labels_
        mask = np.zeros_like(labels, dtype=bool)
        mask[db.core_sample_indices_] = True
        
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # if the mask is almost empty
        if n_clusters_<1:
            return None
        unique_labels = set(labels)

        centers = np.zeros((n_clusters_, collec.shape[1]+5))
        w1 = self.walls['wall1'][-1]
        w2 = self.walls['wall2'][-1]
        theta = self.walls['angle'][-1]
        for k in range(n_clusters_):
            member_mask = labels == k
            member = collec[mask & member_mask]
            centers[k,:2] = np.mean(member, axis=0)
            
            k_pos = pt_theta_dist(centers[k,:2] , theta)
            r1 = (k_pos-w1)/(w2-w1)
            if self.intf_ref is not None:
                centers[k,-5] = abs(r1-self.intf_ref[0])*2
                centers[k,-4] = abs(r1-self.intf_ref[1])*2
            num = np.count_nonzero(member_mask)
            centers[k,-3] = np.linalg.norm(np.std(member, axis=0))
            centers[k,-2] = -num
            centers[k,-1] = k
        
        # use a weighted criterion to select interfaces from potential bound lines
        centers[:,-2] = centers[:,-2]/abs(centers[:,-2].min()) - centers[:,-3]/abs(
            centers[:,-3].max()) + abs(centers[:,1]/img.shape[1]-0.5)
        centers[:,-5] += centers[:,-2]
        centers[:,-4] += centers[:,-2]
        centers0 = centers[centers[:,-5].argsort(), :]
        centers1 = centers[centers[:,-4].argsort(), :]
        centers = centers[centers[:,-2].argsort(), :]
        # determine interface number
        if n_clusters_ == 1:
            # if only 1 interface, determine it's from up or bottom inlet
            # cen = centers[0,:-2].astype(int)
            # up_mean = roi[cen[1]-6:cen[1]-3, cen[0]-1:cen[0]+1].mean()
            # bot_mean = roi[cen[1]+4:cen[1]+7, cen[0]-1:cen[0]+1].mean()
            # if up_mean>bot_mean:
            if centers[0,-5]<centers[0,-4]:
                id_list = [centers[0,-1],-1]
            else:
                id_list = [-1, centers[0,-1]]
            
        else:
            # if more than 1 interfaces, distinguish up & bottom int
            # for j in range(2,n_clusters_):
            #     if abs(centers[1,1]-centers[j,1])<10:
            #         member_mask = labels == j
            #         labels[member_mask] = int(centers[1,-1])
            if centers0[0,-1]!= centers1[0,-1]:
                id1 = centers0[0,-1]
                id2 = centers1[0,-1]
            
            elif centers[0,1]>centers[1,1]:
                id1 = centers[1,-1]
                id2 = centers[0,-1]
            else:
                id1 = centers[0,-1]
                id2 = centers[1,-1]
            
            id_list = [id1, id2]
        # quantify noise
        snr_mask1 = labels == id_list[0] if id_list[0]>=0 else labels==-2
        snr_mask2 = labels == id_list[1] if id_list[1]>=0 else labels==-2
        snr_data1 = collec[snr_mask1]
        snr_data2 = collec[snr_mask2]
        self.SNR = (snr_data1.shape[0]+snr_data2.shape[0])/collec.shape[0]
        self.n_SNR_data = min(snr_data1.shape[0],snr_data2.shape[0])
        
        intfs = []
        p1 = np.array(p1)

        for x,idx in enumerate(id_list):
            if idx<0:
                intfs.append(None)
                continue
            member_mask = labels == idx
            intf0 = collec[mask & member_mask]
            seed = np.random.choice(intf0.shape[0], min(50,intf0.shape[0]), replace = False)
            # intf1 = intf0[seed]*4 + p1
            # intfs.append(intf1)
            intf0 = intf0+p1//4
            # use erosion to thinner the detected point cloud
            tip = np.zeros_like(roi)
            xx = 2-x*4
            for j in range(intf0.shape[0]):
                tip[intf0[j,1]+xx, intf0[j,0]] = 255
            tip = cv2.pyrUp(tip)
            tip = cv2.pyrUp(tip)
            num = cv2.findNonZero(tip).shape[0]
            while 1:
                tip = cv2.erode(tip,np.ones((3,3),np.uint8),iterations = 1)
                if cv2.findNonZero(tip).shape[0]<num*0.9:
                    break
            
            # imgFinal = blend_images(tip, self.last_frame, 1, 0.7)
            # cv2.imshow('final'+str(idx), tip)
            ret,tip = cv2.threshold(tip,180,255,cv2.THRESH_BINARY ) 
            cc = cv2.findNonZero(tip)
            cc = np.squeeze(cc)
            seed = np.random.choice(cc.shape[0], min(200,cc.shape[0]), replace = False)
            intf1 = cc[seed]
            intfs.append(intf1)
            
        # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        # plt.close()
        # for k, col in zip(unique_labels, colors):
        #     if k == -1:
        #         # Black used for noise.
        #         col = [0, 0, 0, 1]
        #         # continue
        
        #     class_member_mask = labels == k
        
        #     xy = collec[class_member_mask & mask]
        #     plt.plot(
        #         xy[:, 0],
        #         -xy[:, 1],
        #         "o",
        #         markerfacecolor=tuple(col),
        #         markeredgecolor="k",
        #         markersize=14,
        #     )
        # plt.title(f"Estimated number of clusters: {n_clusters_}")
        # plt.show()
        
        return intfs

    def hough(self, edged, n=4, th=50, eps=50, ms=2):
        """ 
        Hough Line detect, and cluster resulting lines with DBSCAN to find 
        the most probable lines (walls/interfaces)
        """
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
        if 'local_inter' in self.data[self.cur_idx-self.sd].keys():
            # linep = self.data[self.cur_idx-1]['local_inter']
            linep = np.median(np.array(self.queue)[-21:,:], axis=0)
            linep = np.expand_dims(linep, axis=0)
            if abs((linep-lines)[0,1])>np.pi/4:
                lines = linep.copy()
        return lines
    
    def search_int(self, img, lines, s1, s2):
        """
        s1 : tuple, first point for searching area
        s2 : tuple, second point for searching area
        """
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
        if min(abs(theta),abs(360-theta))>0.5:
            st=s1[0]
            ed=s2[0]
            for dr in range(p1,p2):
                for col in range(st,ed):
                    # print(dr-p1,', ',round((dr-col*np.cos(theta))/np.sin(theta)),', ',col )
                    try:
                        cum[dr-p1] += img[int(round((dr-col*np.cos(theta))/np.sin(theta))),col]
                    except:
                        print(dr-p1,', ',round((dr-col*np.cos(theta))/np.sin(theta)),', ',col )
                cum[dr-p1] = cum[dr-p1]/(ed-st)
        else:
            st=s1[1]
            ed=s2[1]
            for dr in range(p1,p2):
                for row in range(st,ed):
                    cum[dr-p1] += img[row, int(round((dr-row*np.sin(theta))/np.cos(theta)))]
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
                if avg-valleys[1,i]> peaks[1, -i-1]-avg:
                    result.append(candvl+p1)
                    # result.append(candpk+p1)
                else:
                    # result.append(candpk+p1)
                    result.append(candvl+p1)
                if len(result)>=2:
                    break

        return result
        