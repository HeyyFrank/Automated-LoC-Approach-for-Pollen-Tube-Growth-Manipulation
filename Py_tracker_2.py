"""
    Integration of PT tracker into labview.
"""
from threading import Thread
import sys, os
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import copy
from statistics import mean
from easydict import EasyDict as edict
from tkinter import *
from PIL import ImageTk,Image  
from queue import Queue, Empty
import traceback
import pandas as pd

from bg_tracker import bg_Tracker
from MOSSE_tracker import mo_Tracker
from utils import MousePositionTracker, SelectionObject, img_process, bbox2pt, pt_theta_dist
from sklearn.linear_model import RANSACRegressor
from controller import PIDController, KalmanFilter
xrange = range
"""
Golbal variables:
    BF: Bright field (1) or fluo (0)
    Track_means: contour (0) or motion (1) or fluo (2) 
    Resolution: auto (0) or bit size (8/12/16)
    Auto: disable (0) or driving to PT tip (1) or driving to user-selected pt (2)
    cap_mode: online MPTIFF (0) or online TIFF (1) or offline video (2)
"""
BF = 1
Track_means = 1
Resolution = 0
Auto = 0
TFR = 0
flow1 = 0
flow2 = 0
flow3 = 0
cap_mode = 0
cap_path = '' #'C:/My Data/A_Personal/ETH/LabVIEW/test/1/2'
file_name = '' #'Test-0318'

# --- Python LabVIEW essential func ---
def set_value(name, value):
    globals()[name] = value

def get_value(name):
    return globals()[name]

# --- Queue/thread infrastructure ---
status_queue = Queue()
flow_queue = Queue()
flow_queue2 = Queue()
flow_queue3 = Queue()
worker_thread = None

def is_running():
    """ True if the worker thread is running, False otherwise """
    if worker_thread is not None and worker_thread.is_alive():
        return True
    return False

def check_queue():
    """ Get the latest item on the queue, or return an empty string """
    try:
        result = status_queue.get_nowait()
        if result is None:
            result = ""
    except Empty:
        result = ""
    return result

def check_flow():
    """ Get the desired flowrate, or return an empty value """
    try:
        result1 = flow_queue.get_nowait()
        if result1 is None:
            result1 = -1
    except Empty:
        result1 = -1
    return result1

def check_flow2():
    """ Get the desired flowrate, or return an empty value """
    try:
        result2 = flow_queue2.get_nowait()
        if result2 is None:
            result2 = -1
    except Empty:
        result2 = -1
    return result2

def check_flow3():
    """ Get the desired flowrate, or return an empty value """
    try:
        result3 = flow_queue3.get_nowait()
        if result3 is None:
            result3 = -1
    except Empty:
        result3 = -1
    return result3

def set_TFR(v, f1, f2, f3):
    global TFR, flow1, flow2, flow3
    TFR = f1+f2+f3
    flow1 = f1
    flow2 = f2
    flow3 = f3
        
def set_capture_path(p, name):
    global cap_path, file_name
    cap_path = p
    file_name = name

def set_mode(m1, m2, m3, m4, m5):
    global cap_mode, BF, Track_means, Resolution, Auto
    cap_mode = m1
    BF = m2
    Track_means = m3
    Resolution = m4
    Auto = m5

# --- Python and LabVIEW interface code --- 
keep_running = False  # We can set this from LabVIEW to control the behavior
                      # of the loop, and exit early.

def start_tracker():
    """ Simulates a long-running Python operation that periodically puts
    information into the queue.
    """
    status_queue.put("Starting up...")
    global GUI
    
    # config option
    tracker_type='FL'
    case = '0'
    st = 1
    stride = 1
    output_tracking_video = 0
    fps = 10
    BF = 1
    if tracker_type == 'BG':
        bg_means = 1
    elif tracker_type == 'FL':
        bg_means = 2
        BF = 0
    else:
        bg_means = 0

    # selct video
    r = "C:/My Data/A_Personal/ETH/1Exp/PT@CF/"
    r2= "D:/A_Personal/ETH/1Exp/PT@CF/"
    file0 = "YC12_CFP.tif"
    file1 = '2022/0720/video(00001).tif'
    # file2= '0401/video(00000).tif'
    file2= '2022/0829/20mM(0).tif'
    file2= '2022/0928/7.5mM (2).tif'
    file3 = '2022/0224/chip01/video_combine.tif'
    file4 = "2023.01.12/20 mM(0).tif"
    file6 = "2023.05.24/20uM(0).tif"
    file7 = "2023.06.02/20uM(0).tif"
    
    file_path = r + file6
    bit = 8
    dataset = Image.open(file_path) 
    h,w = np.shape(dataset)
        
    # GUI = Display(data=dataset, st=st, stride=stride, bg_means=bg_means, 
    #                mode=2, BF=BF, bit=bit, )
    GUI = Display(cap_path = '')
    
    # # See if LabVIEW has requested that we stop.
    # if not keep_running:
    #     status_queue.put("Exiting early...")

    # # Do work.  In this case, we just sleep for 1 second.
    # status_queue.put("Working... ")
    # time.sleep(1)
        
    status_queue.put("Finished.")


def launch_complicated_operation():
    """ Called from LabVIEW to start the operation in a parallel thread """
    global worker_thread
    
    worker_thread = Thread(target=start_tracker)
    
    # This line is important; the "daemon" flag instructs Python not to wait
    # for the thread to finish before exiting.  We set it to True, so that
    # when we use Close Session from LabVIEW, Python will be sure to exit.
    worker_thread.daemon = True
    worker_thread.start()

class Display:
    def __init__(self, width = 1000, height=750, tracker_type='bg', bg_means=1, 
                 data=None, img=None, out=None, out_file=None, st=0, stride=1,
                 mode=0, cap_path = '', BF=1, bit=12, img_enh=True):
        self.cfg = edict()
        self.cfg.tracker_type = tracker_type
        self.cfg.frame_width = width
        self.cfg.frame_height = height
        self.cfg.st = st
        self.cfg.stride = stride
        self.cfg.bg_means = bg_means
        self.cfg.SELECT_OPTS = dict(dash=(2, 2), stipple='gray25', fill='red', outline='')
        self.cfg.out_file = out_file
        self.cfg.cap_path = cap_path
        self.cfg.cap_mode = mode
        self.cfg.BF = BF
        self.cfg.img_bit = bit
        self.cfg.bit_max_value = 2**self.cfg.img_bit-1
        self.cfg.img_enhance = img_enh
        self.buffer = None

        self.tmp = edict()
        self.tmp.idx = 0
        self.tmp.roi = None
        self.tmp.ibox = None
        self.tmp.inter = False
        self.tmp.intf_data = None
        self.tmp.object2select = ''
        self.tmp.track_status = 0
        self.tmp.live = 1
        self.tmp.video =[]
        self.tmp.data_online = []
        self.tmp.n_buffer = 0
        self.tmp.cur_df = {}
        self.TFR = 0
        
        self.ctrl = edict()
        self.ctrl.set_delay = 2
        self.ctrl.cur_delay = 0
        self.ctrl.gain = 0.3
        self.ctrl.pid_controller = None

        self.res = edict()
        pd_data = {'frame': [],'center_X': [],'center_Y': [], 'angle': [], 'centerline':[],
                'flow_1': [], 'flow_2': [], 'flow_3': [], 'wall_1': [], 'wall_2': [], 'wall_ang': [],
                'pt_pos_1': [], 'state_1_obs': [], 'state_2_obs': [], 'sp':[], 'min_dist':[],
                'x1':[], 'x2':[], 'u1':[], 'u2':[], 'tar1':[], 'tar2':[]}
        self.res.df = pd.DataFrame(pd_data)
        self.res.ep=[]
        self.res.cents=[]
        self.res.out = out
        self.res.ok = False

        if data is not None:
            self.dataset = data
            self.dataset.seek(self.tmp.idx+self.cfg.st)
            img = np.array(self.dataset)
            if self.cfg.img_enhance:
                img = (img/img.max())
            else: 
                img = (img/self.cfg.bit_max_value)
            img = (img*255).astype(np.uint8)
        else:
            img = np.ones((960, 1280),dtype=np.uint8)*255
            self.dataset = None
        self.tmp.first_frame = img.copy()

        self.TLD_tracker = None
        self.init_tracker()
        self.create_interface()
        
    def init_tracker(self):
        if self.cfg.tracker_type == 'bg':
            self.tracker = bg_Tracker(1280, 960)
            
        else:
            self.tracker = mo_Tracker('MOSSE')
            
    def init_controller(self):
        # System parameters
        A = np.array([[1.0, 0], 
                      [0, 1.0]])
        B_est = np.array([[1.0, 0], 
                      [0, 1.0]])/TFR
        C = np.array([[1.0, 0], [0, 1.0]])
        Q = 0.01 * np.eye(2)  # Process noise covariance
        R = 0.01 *20* np.eye(2)  # Measurement noise covariance
        # Initial conditions
        initial_state_estimate = np.array([[0.3], [0.7]])
        initial_state_covariance = np.eye(2)
        self.ctrl.control_input = np.array([[0.0], [0.0]]) 
        # PID parameters
        kp = np.array([[0.1], [0.1]])*25
        ki = np.array([[0.01], [0.01]])*10
        kd = np.array([[0.005], [0.005]])*10
        setpoint = np.array([[0.3], [0.7]])  # Desired setpoint
        
        # Create Kalman filter and PID controller
        self.ctrl.kalman_filter = KalmanFilter(A, B_est, C, Q, R, initial_state_estimate, initial_state_covariance)
        self.ctrl.pid_controller = PIDController(kp, ki, kd, setpoint, self.ctrl.kalman_filter)

               
    def create_interface(self):
        self.root= Tk()
        self.root.title('Pollen Tube Tracker')
        self._job = None
        self._syn = None
        
        img = Image.fromarray(self.tmp.first_frame)
        new_img = self.resize_img(img)
        self.tmp.video.append(new_img)
        
        for row in range(16):
            self.root.rowconfigure(row, pad=5)
        for col in range(5):
            self.root.columnconfigure(col, pad=5)
        
        self.root.geometry(str(self.cfg.frame_width+200)+'x'+str(self.cfg.frame_height+70))
        # display area
        self.canvas = Canvas(self.root, width = self.cfg.frame_width, height = self.cfg.frame_height)  
        self.canvas.grid(row=0,rowspan=15,column=0, pady=5) #place(x=5,y=5)
        
        self.scr = self.canvas.create_image(0, 5, anchor=NW, image=self.tmp.video[-1])
        # instruction area
        self.var_scr = StringVar()
        self.var_scr.set('Please first select region of interest(ROI) to be tracked. Press Select button in ROI Selection.')
        self.text_scr = Label(                                                                                      
            self.root, textvariable= self.var_scr, 
            anchor='nw', height =2, width=110, borderwidth=2, relief="groove").grid(row=15,column=0) #place(
                # x=8, y=self.frame_height+10)
        
        self.loc = StringVar()
        self.loc.set('None, None')
        self.text1 = Label(self.root,text='The object is located at:',).grid(row=0,column=1, columnspan=4) #.place(x=self.frame_width+10, y=10)
        self.text_location = Label(self.root,textvariable=self.loc,width=20,height =3, 
                                   borderwidth=2, relief="groove").grid(row=1,column=1, columnspan=4) #place(
                                       # x=self.frame_width+10, y=40)
        # pixelVirtual = PhotoImage(width=1, height=1)
        Label(self.root,text='ROI Selection:',).grid(row=2,column=1, columnspan=4, sticky='w')#place(x=self.frame_width+10, y=360)
        self.button_roi_pt = Button(self.root,text = 'PT', width= 8, 
                command = self.select_pt_roi).grid(row=3,column=1, columnspan=2)
        self.button_roi_int = Button(self.root,text = 'Interface', width= 8, 
                command = self.select_int_roi).grid(row=3,column=3, columnspan=2)
        self.button_enter = Button(self.root,text = 'Confirm', width= 18, bg='royal blue', fg='white',
                command = self.finish_roi).grid(row=4,column=1, columnspan=4)#place(x=self.frame_width+100, y=400)
        Label(self.root,text='C:').grid(row=5, column=1, columnspan=1)
        Button(self.root,text = '>', width= 4, bg='royal blue', fg='white', 
               command = self.cam_next).grid(row=5,column=2, columnspan=1)
        Button(self.root,text = '▶', width= 4,  bg='forest green', fg='white',
               command = self.cam_live).grid(row=5,column=3, columnspan=1)
        Button(self.root,text = '■', width= 4, bg='crimson', fg='white',
                command = self.cancel).grid(row=5,column=4, columnspan=1)
        
        Label(self.root,text='Tracking:',).grid(row=6,column=1, columnspan=4, sticky='w')
        self.button2 = Button(self.root,text = '>', width= 8, bg='royal blue', fg='white',
                command = self.forward).grid(row=7,column=1, columnspan=2)#place(x=self.frame_width+10, y=220)
        # Label(self.root,text='Auto Tracking:',).grid(row=6,column=1, columnspan=4, sticky='w')#place(x=self.frame_width+10, y=280)
        self.button3 = Button(self.root,text = '▶', width= 4, bg='forest green', fg='white',
                command = self.auto_track).grid(row=7,column=3, columnspan=1)#place(x=self.frame_width+10, y=310)
        self.button4 = Button(self.root,text = '■', width= 4, bg='crimson', fg='white',
                command = self.cancel).grid(row=7,column=4, columnspan=1)#place(x=self.frame_width+100, y=310)
        Label(self.root,text='Interface:',).grid(row=8,column=1, columnspan=2, sticky='w')
        self.button_inter_on = Button(self.root,text = 'On', width= 4,  bg='royal blue', fg='white',
                command = self.interface_on).grid(row=8, column=3, columnspan=1)
        self.button_inter_off = Button(self.root,text = 'Off', width= 4, 
                command = self.interface_off).grid(row=8, column=4, columnspan=1)
        Button(self.root,text = 'Restart', width= 8, 
                command = self.restart).grid(row=9,column=1, columnspan=2)
        Button(self.root,text = 'Global PT', width= 8, 
                command = self.global_PT).grid(row=9,column=3, columnspan=2)
        # input area
        self.var1=StringVar()
        self.var2=StringVar()
        self.var1.set('')
        self.var2.set('')
        self.text2 = Label(self.root,text='Select location to drive:', anchor='nw',justify=LEFT,
              width=20,  wraplength=180).grid(row=10,rowspan=1, column=1, columnspan=4)#.place(x=self.frame_width+10, y=70)
        Label(self.root, text='Interface to drive:',).grid(row=11,column=1, columnspan=3)#.place(x=self.frame_width+10, y=115)
        self.entry1 = Entry(self.root,textvariable=self.var1,bd=1, width=4).grid(row=11,column=4, )#.place(x=self.frame_width+30, y=115)
        # Label(self.root, text='y:',).grid(row=11,column=3, )#.place(x=self.frame_width+110, y=115)
        # self.entry2 = Entry(self.root,textvariable=self.var2,bd=1,width=6).grid(row=11,column=4, )#.place(x=self.frame_width+130, y=115)
        self.button1 = Button(self.root,text = 'Drive', width= 8, 
               command = self.show_msg).grid(row=12,column=1, columnspan=2)#place(x=self.frame_width+10, y=145)
        Button(self.root,text = 'Stop', width= 8, bg='crimson', fg='white',
               command = self.stop_drive).grid(row=12,column=3, columnspan=2)
        Button(self.root,text = 'Check', width= 8, 
                command = self.check).grid(row=13,column=1, columnspan=2)
        Button(self.root,text = 'Transfer', width= 8, 
                command = self.transfer).grid(row=13,column=3, columnspan=2)
        Button(self.root,text = 'Sync', width= 8, 
                command = self.sync).grid(row=14,column=1, columnspan=2)
        Button(self.root,text = 'Cancel Sync', width= 8, 
                command = self.cancel_syn).grid(row=14,column=3, columnspan=2)
        Button(self.root,text = 'Save', width= 18, 
                command = self.save).grid(row=15,column=1, columnspan=4)
        
        mainloop()
        self.save()
    
    def resize_img(self, img):
        major_ver, minor_ver, subminor_ver = (Image.__version__).split('.')
        major_ver=int(major_ver)
        if major_ver<9:
            resized_img= img.resize((self.cfg.frame_width, self.cfg.frame_height), Image.ANTIALIAS)
        else:
            resized_img= img.resize((self.cfg.frame_width, self.cfg.frame_height), Image.Resampling.LANCZOS)
        new_img= ImageTk.PhotoImage(resized_img)
        return new_img
    
    def read_next_seq(self):
        new, img = 0, None
        if self.cfg.cap_mode==2 and self.tmp.idx+self.cfg.st < self.dataset.n_frames-self.cfg.stride :
            self.tmp.idx+=self.cfg.stride
            self.dataset.seek(self.tmp.idx+self.cfg.st)
            img = np.array(self.dataset)
            if self.cfg.img_enhance:
                img = (img/img.max())
            else: 
                img = (img/self.cfg.bit_max_value)
            img = (img*255).astype(np.uint8)
            new = 1
        elif self.cfg.cap_mode==0:
            self.buffer = Image.open(self.cfg.file_name) 
            try:
                buffer_frames = self.buffer.n_frames
                if buffer_frames > self.tmp.n_buffer:
                    self.tmp.n_buffer = buffer_frames
                    self.tmp.idx = buffer_frames-1
                    self.buffer.seek(self.tmp.idx)
                    img = np.array(self.buffer)
                    if self.cfg.img_enhance:
                        img = (img/img.max())
                    else: 
                        img = (img/self.cfg.bit_max_value)
                    img = (img*255).astype(np.uint8)
                    new = 1
            except TypeError:
                new = 0
        elif self.cfg.cap_mode==1: 
            new, img = self.cap_path_check()
        return new, img

    def cam_next(self):
        new, img = self.read_next_seq()
        if new>0:
            
            img = Image.fromarray(img)
            new_img2= self.resize_img(img)
            self.tmp.video[-1]=new_img2
            self.update_show()
    
    def cam_live(self):
        if self.tmp.live:
            if self.cfg.cap_mode <2 or self.tmp.idx+self.cfg.st < self.dataset.n_frames-self.cfg.stride :
                self.cam_next()
                self._job = self.root.after(20, self.cam_live)
            else:
                self.cancel()
    
    def restart(self):
        """ Restart tracker, reset ROI """
        self.tmp.roi = None
        self.tmp.inter = False
        self.tmp.object2select = ''
        self.tmp.track_status = 0
        self.tmp.live = 1
        self.tmp.video =[]
        self.res.ok = False
        self.init_tracker()

        if self.cfg.cap_path =='':
            self.dataset.seek(self.tmp.idx+self.cfg.st)
            img = np.array(self.dataset)
            
        else:
            if self.buffer is not None:
                self.buffer = Image.open(self.cfg.file_name) 
                self.tmp.n_buffer = self.buffer.n_frames
                self.buffer.seek(self.tmp.n_buffer-1)
                img = np.array(self.buffer)
            else:
                l = sorted(os.listdir(self.cfg.cap_path))
                img = Image.open(self.cfg.cap_path+'/'+l[-1])
                img = np.array(img)

        if self.cfg.img_enhance:
            img = (img/img.max())
        else: 
            img = (img/self.cfg.bit_max_value)
        img = (img*255).astype(np.uint8)
        self.tmp.first_frame = img.copy()
        img = Image.fromarray(img_process(img))
        new_img = self.resize_img(img)
        self.tmp.video.append(new_img)
        self.update_show()
        
    def global_PT(self):
        """ Find PT globally """
        if self.TLD_tracker is None:
            self.TLD_tracker = cv2.TrackerMOSSE_create() 
            # self.TLD_tracker = cv2.TrackerTLD_create()
            # self.tmp.tracking_status = self.TLD_tracker.init_tracker(0, self.tmp.first_frame, tuple(self.tmp.bbox), self.cfg.stride)
            frame = img_process(self.tmp.first_frame)
            ok = self.TLD_tracker.init(frame, tuple(self.tmp.bbox))
        
        if self.cfg.cap_path =='':
            self.dataset.seek(self.tmp.idx+self.cfg.st)
            img = np.array(self.dataset)
            
        else:
            if self.buffer is not None:
                self.buffer = Image.open(self.cfg.file_name) 
                self.tmp.n_buffer = self.buffer.n_frames
                self.buffer.seek(self.tmp.n_buffer-1)
                img = np.array(self.buffer)
            else:
                l = sorted(os.listdir(self.cfg.cap_path))
                img = Image.open(self.cfg.cap_path+'/'+l[-1])
                img = np.array(img)

        if self.cfg.img_enhance:
            img = (img/img.max())
        else: 
            img = (img/self.cfg.bit_max_value)
        img = (img*255).astype(np.uint8)
        img = img_process(img)
        ok, bbox = self.TLD_tracker.update(img)
        p1, p2 = bbox2pt(bbox)
        cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
        img = Image.fromarray(img)
        new_img2= self.resize_img(img)
        self.tmp.video[-1]=new_img2
        self.update_show()
        self.var_scr.set('Detection complete.')

                
    def mouse(self, event):
        if self.mouse_enable:
            x,y = event.x, event.y
            self.var_scr.set('Selected point: {}, {}'.format(x, y))
        
    def click(self, event):
        if self.mouse_enable:
            global Auto
            x,y = event.x, event.y
            self.var_scr.set('Selected point: {}, {} confirmed, driving there...'.format(x, y))
            self.mouse_enable = False
            Auto = 2
            self.tmp.drive_p = (x,y)
            self.drive((x,y))
        
    def show_msg(self):
        if self.res.ok:
            # if self.var1.get() !='' and self.var2.get() !='':
            #     global Auto
            #     self.var_scr.set('Input grid: {}, {} confirmed, driving there...'.format(self.var1.get(),self.var2.get()))
            #     Auto = 2
            #     self.tmp.drive_p = (self.var1.get(),self.var2.get())
            #     self.drive(self.tmp.drive_p)
            # else:
            self.var_scr.set('No input detected. Please select the point by clicking in the frame.')
            self.mouse_enable = True
            self.canvas.bind('<Motion>', self.mouse)
            self.canvas.bind("<Button-1>", self.click)

    def drive(self, p, sp=-1, th=15):
        """ 
        Args:
            p [tuple (x,y)]: driving point, user-decided or PT tip
            sp [int]: driven interface, upper (0) or bottom (1) or self-decide (-1 when user-select)
            th [pixel]: threshold within which distance to stop controller
        Var:
            over_sign: overshoot (>0) or not (<0)
        """
        global Auto
        if not Auto:
            return None
        try:
            # check if tracking is enabled and normal
            pos = self.tmp.pos
        except:
            return None
        if sp<0:
            # user defined pt
            th = 0
            center = self.tracker.center
            if center[1]<p[1]:
                sp=1
            else:
                sp=0
        # init controller if not yet
        if self.ctrl.pid_controller is None:
            self.init_controller()
        # handle delay
        if self.ctrl.cur_delay%self.ctrl.set_delay>0:
            self.ctrl.cur_delay+=1
            return None
        else:
            self.ctrl.cur_delay = 1
        try:
            # check if the interface is detected
            intf = self.tracker.data[self.tmp.idx]['inter'][sp]
            p = np.array(p)
            # 1st horizontal position
            n = 2*self.tracker.roi[1]-self.tracker.roi[0]
            intf_y = intf[((intf[:,0]>self.tracker.roi[1])&(intf[:,0]<n))]
            while intf_y.shape[0]<1 and n < np.shape(self.tmp.first_frame)[1]:
                n += self.tracker.roi[1]-self.tracker.roi[0]
                intf_y = intf[((intf[:,0]>self.tracker.roi[1])&(intf[:,0]<n))]
                
            y_center = np.mean(intf_y, axis=0)
            if sp:
                intf_y = intf_y[(intf_y[:,1]<y_center[1])]
            else:
                intf_y = intf_y[(intf_y[:,1]>y_center[1])]
            y_center = np.mean(intf_y, axis=0)
            theta = pos[3]
            dist_min_y = pt_theta_dist(p, theta) - pt_theta_dist(y_center, theta)
            rPos = np.sign(dist_min_y)
            dist_min_y = abs(dist_min_y)
            
            # 2nd mim. distance
            intf_local = intf[((intf[:,0]>self.tracker.roi[0])&(intf[:,0]<self.tracker.roi[1]))]
            if intf_local.size>0:
                # dist_ave = np.mean(intf_local, axis=0)
                # dist_std = np.std(intf, axis=0)
                # print(sp, dist_ave, p, dist_std)
                # if abs(p[1]-dist_ave[1])<15 and dist_std[1]<5:
                #     return 0
                dist_local = np.linalg.norm(intf_local - p, axis = 1)
                dist_min_local = min(dist_local) 
            else:
                dist_min_local = 1000
            dist_min = min(dist_min_y, dist_min_local)
            
            # print("sp={:d}, min_y={:02.2f}, min_local={:02.2f}, ori={:.1f}".format(sp, dist_min_local, dist_min_y, rPos))
            
            over_sign = rPos*(sp-0.5)
            self.tmp.cur_df['sp'] = sp
            self.tmp.cur_df['min_dist'] = dist_min*rPos
            self.tmp.cur_df['tar'+str(sp+1)] = pos[0]
            self.tmp.cur_df['tar'+str(2-sp)] = 0.3*pos[0] + 0.7*(1-sp)
            self.tmp.cur_df['x'+str(sp+1)] = pos[0]-dist_min*rPos/(self.tmp.cur_df['wall_2']-self.tmp.cur_df['wall_1'])
            process_variable = np.array([self.tmp.cur_df['x1'],self.tmp.cur_df['x2']]).reshape((2,1))
            setpoint = np.array([self.tmp.cur_df['tar1'],self.tmp.cur_df['tar2']]).reshape((2,1))
            control_output = self.ctrl.pid_controller.compute(process_variable, self.ctrl.control_input, setpoint)
            
            # if dist_min<=th and over_sign<0:
            #     self.ctrl.control_input = np.array([[0.0], [0.0]]) 
            #     # return None
            # else:
            # safety check
            control_output[0,0] = np.clip(control_output[0,0], -flow3, flow1+flow2)
            control_output[1,0] = np.clip(control_output[1,0], -(flow2+flow3), flow1)
            self.ctrl.control_input = control_output
            
            self.tmp.cur_df['u1'] = self.ctrl.control_input[0,0]
            self.tmp.cur_df['u2'] = self.ctrl.control_input[1,0]
            self.TFR = flow1+flow2+flow3
            flow_queue.put(flow1-control_output[1,0])
            flow_queue2.put(flow2+control_output[1,0]-control_output[0,0])
            flow_queue3.put(control_output[0,0]+flow3)
            time.sleep(1)
            
            # # TODO: tune the gain or use learning method
            # flow_change = (dist_min-th)/(pos[2])*self.ctrl.gain
            # over_change = (dist_min+th)/(pos[2])*self.ctrl.gain
            # self.TFR = flow1+flow2+flow3
            # if sp:
            #     if rPos>0:  # overshoot p[1]-dist_ave[1]
            #         x = min(flow1-over_change*self.TFR, (1-pos[0]*0.3)*self.TFR)
            #     else:
            #         x = min(flow1+flow_change*self.TFR, (1-pos[0]*0.3)*self.TFR)
            #     flow_queue.put(x)
            #     flow_queue2.put(self.TFR-x-pos[0]*self.TFR*0.3)
            #     flow_queue3.put(pos[0]*self.TFR*0.3)
            #     time.sleep(1)

            # else:
            #     if rPos>0:
            #         x = min(flow3+flow_change*self.TFR, (1-pos[1]*0.3)*self.TFR)
            #     else: # overshoot
            #         x = min(flow3-over_change*self.TFR, (1-pos[1]*0.3)*self.TFR)
            #     flow_queue2.put(self.TFR-x-pos[1]*self.TFR*0.3)
            #     flow_queue.put(pos[1]*self.TFR*0.3)
            #     flow_queue3.put(x)
            #     time.sleep(1)

        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info
            self.ctrl.cur_delay = 0
            # if not detected, use lookup table instead of close loop
            # if sp:
            #     # flow_change = pos[1]*self.TFR - flow1+pos[0]*self.TFR*0.5-flow3
            #     flow_queue.put(pos[1]*self.TFR)
            #     flow_queue2.put((1-pos[1]-pos[0]*0.5)*self.TFR)
            #     flow_queue3.put(pos[0]*self.TFR*0.5)
            #     time.sleep(1)
            # else:
            #     # flow_change = pos[0]*self.TFR - flow3
            #     flow_queue.put(pos[1]*self.TFR*0.5)
            #     flow_queue2.put((1-pos[0]-pos[1]*0.5)*self.TFR)
            #     flow_queue3.put(pos[0]*self.TFR)
            #     time.sleep(1)
        
        # print("f1={:02.3f}, f2={:02.3f}, f3={:02.3f}".format(flow_queue.get_nowait(), 
        #                                                      flow_queue2.get_nowait(), 
        #                                                      flow_queue3.get_nowait()
        #                                                      )
        #       )
        # status_queue.put("Flow 1 = {:.2f}, Flow 2 = {:.2f} (ul/min)".format(f1, f2))
        
    def stop_drive(self):
        global Auto
        self.var_scr.set('Stop driving.')
        Auto = 0
        
    def update_show(self):
        self.canvas.itemconfig(self.scr, image=self.tmp.video[-1])
        # cv2.imshow("Tracking", self.tmp.video[-1])
        
    def forward(self):
        if self.res.ok:
            # check next frame and update cur idx
            new, img = self.read_next_seq()
            if new>0:
                self.tmp.cur_df['frame'] = self.tmp.idx
                if self.cfg.tracker_type == 'bg':
                    intf_ref = (flow3/(flow1+flow2+flow3+0.00001), (flow2+flow3)/(flow1+flow2+flow3+0.00001))
                    if self.var1.get()=='9':
                        intf_reg = 'fixed'
                    else:
                        intf_reg = 'auto'
                    if self.cfg.BF:
                        img, pos =self.tracker.track(self.tmp.idx, img, self.tmp.inter, self.tmp.ibox, self.cfg.bg_means)
                    else:
                        img, pos =self.tracker.track(self.tmp.idx, img, self.tmp.inter, self.tmp.ibox, 2, intf_ref, intf_reg)
                        centerline = self.tracker.data[self.tmp.idx]['centerline_global'] 
                        self.tmp.cur_df['centerline'] = centerline
                    self.tmp.pos = pos
                    if self.res.out is not None:
                        self.res.out.write(img)
                    gamma = self.tracker.gamma
                    center = self.tracker.center
                    self.tmp.cur_df['center_X'] = center[0]
                    self.tmp.cur_df['center_Y'] = center[1]
                    try:
                        ang = np.arctan(gamma[1]/gamma[0])/np.pi*180
                        self.tmp.cur_df['angle'] = ang
                        self.loc.set('Center: {:d}, {:d}; \nAngle: {:.1f}'.format(
                                self.tracker.center[0],self.tracker.center[1], ang))
                        self.tmp.cur_df['flow_1'] = flow1
                        self.tmp.cur_df['flow_2'] = flow2
                        self.tmp.cur_df['flow_3'] = flow3
                    
                        if self.tmp.inter: #and abs(np.dot(gamma, np.array([1,0])))<0.86
                            try:
                                self.tmp.cur_df['pt_pos_1'] = pos[0]
                                self.tmp.cur_df['wall_ang'] = pos[3]
                                self.tmp.cur_df['wall_1'] = self.tracker.walls['wall1'][-1]
                                self.tmp.cur_df['wall_2'] = self.tracker.walls['wall2'][-1]
                                self.save_state_obs()
                            except:
                                pass
                            global Auto
                            if Auto>1:
                                self.drive(self.tmp.drive_p)
                            elif Auto>0:
                                # decide which interface to drive based on orientation
                                if self.var1.get() in ['0', '1']:
                                    sp = int(self.var1.get())
                                    self.drive(center, sp)
                                    self.tmp.intf_data = [self.tmp.idx, sp, center, ang]
                                    self.var1.set('')
                                    
                                elif self.tmp.intf_data is None:
                                    wall_angle = pos[3]
                                    print(ang, wall_angle/np.pi*180)
                                    sp = int(np.sign(90+ang-wall_angle/np.pi*180)*0.5+0.5)
                                    ratio = pos[0]
                                    if (sp-0.5)*(ratio-0.5)>0:
                                        self.drive(center, sp)
                                        self.tmp.intf_data = [self.tmp.idx, sp, center, ang]
                                        
                                else: 
                                    # tunable
                                    dist = np.linalg.norm(center-self.tmp.intf_data[2])*0.77
                                    if abs(ang-self.tmp.intf_data[3])>15 and dist>100:
                                        wall_angle = pos[3]
                                        sp = int(np.sign(90-ang-wall_angle/np.pi*180)*0.5+0.5)
                                        self.drive(center, sp)
                                        self.tmp.intf_data = [self.tmp.idx, sp, center, ang]
                                    else:
                                        self.drive(center, self.tmp.intf_data[1])
                    except Exception:
                        exc_info = sys.exc_info()
                        traceback.print_exception(*exc_info)
                        del exc_info
                        print("Error!")
                    
                else:
                    ok, img, p1, p2 = self.tracker.update(self.tmp.idx, img, self.tmp.inter, self.tmp.ibox)
                    if ok:
                        if self.res.out is not None:
                            self.res.out.write(img)
                        if self.tracker.bp_ellip is not None:
                            self.res.ep.append([self.tmp.idx, self.tracker.bp_ellip[2]])
                            self.loc.set('Center: {}, {}; \nAngle: {}'.format(
                                str((p1[0]+p2[0])//2),str((p1[1]+p2[1])//2), str(round(270-self.tracker.bp_ellip[2]))))
                            if self.tmp.inter:
                                ibox = list(self.tmp.ibox)
                                ibox[0] = self.tracker.bp_ellip[0][0] - np.sign(self.tracker.bp_ellip[0][0]-(p1[0]+p2[0])//2)*(p2[0]-p1[0])
                                ibox[2] += self.tmp.ibox[0] - ibox[0]
                                self.tmp.ibox = ibox
                    else:
                        self.var_scr.set('Detection failure. Please select ROI again.')
                        self.res.ok = False
                img = Image.fromarray(img)
                new_img2= self.resize_img(img)
                self.tmp.video[-1]=new_img2
                self.update_show()
                self.var_scr.set('Detection complete.')
                self.res.df = self.res.df.append(self.tmp.cur_df, ignore_index=True)
                # if self.TFR>0:
                #     self.set_flow(pos)
        else:
            self.var_scr.set('No ROI is selected. Please select ROI again.')
        
    def auto_track(self):
        if self.res.ok:
            if self.cfg.cap_mode <2 or (self.cfg.cap_mode ==2 and self.tmp.idx+self.cfg.st < self.dataset.n_frames-self.cfg.stride):
                self.forward()
                self._job = self.root.after(5, self.auto_track)
            else:
                self.cancel()

    def cancel(self):
        if self._job is not None:
            self.root.after_cancel(self._job)
            self._job = None
            
    def select_pt_roi(self):     
        self.tmp.object2select = 'pt'
         # Create selection object to show current selection boundaries.
        self.tmp.roi = SelectionObject(self.canvas, self.cfg.SELECT_OPTS)
        self.posn_tracker = MousePositionTracker(self.canvas)
        # Callback function to update it given two points of its diagonal.
        def on_drag(start, end, **kwarg):  # Must accept these arguments.
            if self.posn_tracker.enable:
                self.tmp.roi.update(start, end)
        
        self.posn_tracker.autodraw(command=on_drag)  # Enable callbacks.
        self.var_scr.set('Press Confirm button to confirm selection.')
    
    def finish_roi(self):
        if self.tmp.roi is not None:
            bbox = self.tmp.roi.get_roi()
        else:
            bbox = None

        if bbox is not None:
            self.var_scr.set('ROI confirmed. You can start auto tracking now.')
            self.canvas.delete(ALL)
            
            # self.roi = None
            # self.roi.hide()
            # self.canvas.itemconfig(self.scr, image=self.video[-1])
            # self.roi = self.posn_tracker = None
            
            self.posn_tracker.enable = False
            self.scr = self.canvas.create_image(0, 5, anchor=NW, image=self.tmp.video[-1])
            if self.tmp.object2select == 'pt':
                
                if self.res.ok:
                    # reset roi in tracker/ reset tracker
                    self.restart()
                    
                self.res.ok = True
                h,w = np.shape(self.tmp.first_frame)
                self.tmp.bbox = (int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h))
                self.tmp.tracking_status = self.tracker.init_tracker(
                    self.tmp.idx, self.tmp.first_frame, self.tmp.bbox, 
                    self.cfg.stride, self.cfg.BF, self.cfg.bg_means)
                
            elif self.tmp.object2select == 'int':
                h,w = np.shape(self.tmp.first_frame)
                self.tmp.ibox = (int(bbox[0]*w), int(bbox[1]*h), min(
                    int(bbox[2]*w), w-int(bbox[0]*w)-1), min(int(bbox[3]*h),h-int(bbox[1]*h)-1))
            self.tmp.object2select = ''
                
        else:
            self.var_scr.set('No ROI is selected. Please select ROI again.')
    

    def interface_on(self):
        if self.tmp.ibox is None:
            self.var_scr.set('You haven\'t selected ROI for interface detection.')
            # self.ibox = (768,0,512-1,960-1)
        else:
            self.tmp.inter = True
            if self.cfg.cap_mode==2:
                self.tmp.idx -= self.cfg.stride
            elif self.cfg.cap_mode==0:
                self.tmp.n_buffer -= 1
            self.forward()
            # img = self.tracker.add_inter(self.ibox)
            # new_img2= self.resize_img(img)
            # self.video[-1] = new_img2
            # self.update_show()
    
    def interface_off(self):
        self.tmp.inter = False
    
    def select_int_roi(self):     
        self.tmp.object2select = 'int'
        # Create selection object to show current selection boundaries.
        self.tmp.roi = SelectionObject(self.canvas, self.cfg.SELECT_OPTS)
        self.posn_tracker = MousePositionTracker(self.canvas)
        # Callback function to update it given two points of its diagonal.
        def on_drag(start, end, **kwarg):  # Must accept these arguments.
            if self.posn_tracker.enable:
                self.tmp.roi.update(start, end)
        
        self.posn_tracker.autodraw(command=on_drag)  # Enable callbacks.
        self.var_scr.set('Press Confirm button to confirm selection.')
   
    def save_state_obs(self):
        """ Save state data for further analysis """
        i = 0
        for intf in self.tracker.far_intfs:
            i+=1
            if intf is None:
                continue
            
            y_center = np.mean(intf, axis=0)
            if i>1:
                intf_y = intf[(intf[:,1]<y_center[1])]
            else:
                intf_y = intf[(intf[:,1]>y_center[1])]
            y_center = np.mean(intf_y, axis=0)
            theta = self.tmp.pos[3]
            state_obs = pt_theta_dist(y_center, theta) - self.tmp.cur_df['wall_1']
            self.tmp.cur_df['state_'+str(i)+'_obs'] = state_obs
            self.tmp.cur_df['x'+str(i)] = state_obs/(self.tmp.cur_df['wall_2']-self.tmp.cur_df['wall_1'])
            
        
    def save(self):
        if self.cfg.out_file is not None:
            self.res.df.to_csv(self.cfg.out_file, index=False)
            print("Angle & position data saved.")
        if self.res.out is not None:
            self.res.out.release() 
            print("Tracking video saved.")

    def check(self):
        # See if LabVIEW has requested that we stop.
        if not keep_running:
            status_queue.put("Stopped...")
    
        else:
            # Do work.  In this case, we just sleep for 1 second.
            status_queue.put("Working... ")
            global TFR, cap_path, file_name, cap_mode, BF, Track_means, Resolution
            self.cfg.cap_mode = cap_mode
            self.cfg.BF = BF
            self.cfg.bg_means = Track_means
            if Resolution==0:
                self.cfg.img_bit = Resolution
                self.cfg.img_enhance = True
            else:
                self.cfg.img_enhance = False
                if Resolution == 1:
                    self.cfg.img_bit = 8
                elif Resolution == 2:
                    self.cfg.img_bit = 12
                else:
                    self.cfg.img_bit = 16
                self.cfg.bit_max_value = 2**self.cfg.img_bit-1
            if TFR>0:
                self.TFR = TFR
                status_queue.put("TFR = {} ul/min ".format(self.TFR))
            new=0
            if self.cfg.cap_mode==2:
                status_queue.put("Using offline sequence. ")

            elif self.cfg.cap_mode==1:
                status_queue.put("Using online TIFF. ")
                if cap_path != '':
                    self.cfg.cap_path = cap_path
                    status_queue.put("Capture path successfully set. ")
                    new, img = self.cap_path_check()
                else:
                    status_queue.put("Capture path not set. ")

            elif self.cfg.cap_mode==0:
                status_queue.put("Using online MPTIFF. ")
                if cap_path != '' and file_name !='':
                    self.cfg.cap_path = cap_path
                    if file_name[-4:] != '.tif':
                        file_name = file_name + '.tif'
                    self.cfg.file_name = cap_path+'/'+file_name
                    try:
                        self.buffer = Image.open(self.cfg.file_name) 
                        self.tmp.n_buffer = self.buffer.n_frames
                        self.tmp.idx = self.tmp.n_buffer-1
                        self.buffer.seek(self.tmp.idx)
                        img = np.array(self.buffer)
                        if self.cfg.img_enhance:
                            img = (img/img.max())
                        else: 
                            img = (img/self.cfg.bit_max_value)
                        img = (img*255).astype(np.uint8)
                        new = 1
                        # set up data saving options
                        timestr = time.strftime("%Y%m%d_%H%M_")
                        self.cfg.out_file = cap_path+'/'+timestr+file_name[:-4]+'.csv'
                        h,w = np.shape(img)
                        out_path = cap_path+'/'+timestr+file_name[:-4]+'.avi'
                        self.res.out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (w,h), False)
                    except:
                        status_queue.put("File not found. Please check file path.")
                else:
                    status_queue.put("File path not set. ")
            if new:
                self.tmp.first_frame = img.copy()
                img = Image.fromarray(img)
                new_img2= self.resize_img(img)
                self.tmp.video[-1] = new_img2
                self.update_show()

    def sync(self):
        """ Synchronize between LabVIEW and Python. """
        global TFR, cap_path, file_name, cap_mode
        if keep_running:
            self.cfg.cap_mode = cap_mode
            if TFR>0:
                self.TFR = TFR
            else:
                self.TRF = flow1+flow2+flow3
            # if cap_path != '':
            #     self.cfg.cap_path = cap_path
            # if file_name !='':
            #     if file_name[-4:] != '.tif':
            #         file_name = file_name + '.tif'
            #     self.cfg.file_name = cap_path+'/'+file_name
            self._syn = self.root.after(5000, self.sync)
        else:
            self.cancel_syn()

    def cancel_syn(self):
        if self._syn is not None:
            self.root.after_cancel(self._syn)
            self._syn = None

    def cap_path_check(self):
        l = sorted(os.listdir(self.cfg.cap_path))
        if l and (l[-1] not in self.tmp.data_online):
            self.tmp.data_online.append(l[-1])
            self.tmp.idx = int(l[-1].split('_')[-1][:-4])
            img = Image.open(self.cfg.cap_path+'/'+l[-1])
            img = np.array(img)
            if self.cfg.img_enhance:
                img = (img/img.max())
            else: 
                img = (img/self.cfg.bit_max_value)
            img = (img*255).astype(np.uint8)
            return True, img
        else:
            return False, None

    def transfer(self):
        # Transfer data to LabVIEW
        if not keep_running:
            status_queue.put("Tracker stopped...")
    
        else:
            # Do work. 
            # status_queue.put("Current angle = {} deg".format(
            #     str(round(270-self.tracker.bp_ellip[2]))))
            status_queue.put("Start transfer...")
            time.sleep(1)
            for i in range(1,5):
                status_queue.put("Flow 1 = {}, Flow 2 = {} (ul/min)".format(i*100, (5-i)*100))
                flow_queue.put(i*100/1000)
                flow_queue2.put((5-i)*100/1000)
                time.sleep(3)
    
    def set_flow(self, pos):
        if pos:
            f2 = round(self.TFR*pos[0], 4)
            f1 = self.TFR - f2
            if (self.tmp.idx//self.cfg.stride)% 5 == 0:
                status_queue.put("Flow 1 = {:.2f}, Flow 2 = {:.2f} (ul/min)".format(f1, f2))
            flow_queue.put(f1)
            flow_queue2.put(f2)
            time.sleep(1)


    def set_TFR(self, TFR):
        # Dummy function
        if not keep_running:
            status_queue.put("Stopped...")
    
        else:
            # Do work.  In this case, we just sleep for 1 second.
            self.TFR = TFR
            status_queue.put("Total flow rate is set {} ul/min.".format(TFR))
            

if __name__ == '__main__' :
    # config option
    tracker_type='FL'
    case = '0'
    st = 90
    stride = 1
    output_tracking_video = 0
    fps = 10
    BF = 1
    if tracker_type == 'BG':
        bg_means = 1
    elif tracker_type == 'FL':
        bg_means = 2
        BF = 0
    else:
        bg_means = 0

    # selct video
    r = "C:/My Data/A_Personal/ETH/1Exp/PT@CF/"
    r2= "D:/A_Personal/ETH/1Exp/PT@CF/"
    # FL
    file5 = '2022/1028/10mM + Fluo3 (0).tif'
    # file6 = "2023.05.16/5uM(5).tif"
    file6 = "2023.05.24/20uM(0).tif"
    file7 = "2023.06.02/20uM(3).tif"
    file8 = "2023.07.06/20mM(1).tif"
    file10 = "2023.08.17/Ca20mM(4).tif"
    file11 = "2023.09.08/20mM(0).tif"
    file12 = "2023.09.29/20mM(0).tif"
    file13 = "2023.10.11/20mM(0).tif"
    file14 = "2023/2023.10.31/20mM(4).tif"
    # BF
    file0 = '2022/0720/video(00001).tif'
    file1= '2022/0829/20mM(0).tif'
    file2= '2022/0928/7.5mM (2).tif'
    file3 = '2022/0224/chip01/video_combine.tif'
    file4 = "2023.01.12/20 mM(0).tif"
    file9 = "2023.07.10/Mg130uM(0).tif"
    file15 = "2022/0914/7.5mM lily(0).tif"
    
    file_path = r + file14
    bit = 12
    dataset = Image.open(file_path) 
    h,w = np.shape(dataset)
    timestr = time.strftime("%Y%m%d_%H%M_")

    # set up video
    if output_tracking_video:
        out_path = 'output/'+timestr+tracker_type+'_'+file_path.split('/')[-2]+'_'+file_path.split('/')[-1][:-4]+'.avi'
        out_vid = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h), False)
    else:
        out_vid = None    
    # set up output file
    out_file = 'output/'+timestr+tracker_type+'_'+file_path.split('/')[-2]+'_'+case+'.csv'
    GUI = Display(data=dataset, out=out_vid, out_file=out_file, st=st, 
                  stride=stride, bg_means=bg_means, mode=2, BF=BF, bit=bit, img_enh=True
                  )
    # GUI = Display()
