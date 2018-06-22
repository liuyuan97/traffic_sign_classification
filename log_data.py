
# coding: utf-8

# This class is to define a class to read and process the log data.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class log_data:
    def __init__(self, file_name):        
        camera_dist_scale = 2.0
        
        # read the log data file
        self.log_data = pd.read_csv(file_name, delim_whitespace = True)
        
        # obtain the data field
        self.time = self.log_data['time']
        
        # obtain the host car self position data
        self.host_lat   = self.log_data['l_lat']
        self.host_lon   = self.log_data['l_lon'] 
        self.host_ele   = self.log_data['l_ele']
        self.host_hea   = self.log_data['l_head']
        self.host_speed = self.log_data['l_spe']
        
        # obtain the V2V position data
        self.target_lat   = self.log_data['b_lat']
        self.target_lon   = self.log_data['b_lon'] 
        self.target_ele   = self.log_data['b_ele']
        self.target_hea   = self.log_data['b_head']   
        self.target_speed = self.log_data['b_spe']   
        self.v2v_new_data = self.log_data['b_new']
        
        # get the camera data.  only one car data for now
        self.camera_det_num  = self.log_data['c_num']  # camera detected car numbers
        self.camera_c0_dis   = self.log_data['c0_dis'] * camera_dist_scale
        self.camera_c0_angle = self.log_data['c0_ang'] 
        self.camera_new_data = self.log_data['c_new'] 
        
        # arrange the data to be the right format
        self.arrange_dat()
        self.comp_ned()
        
        # compute the relative distance
        self.relative_dis   = np.sqrt(np.power(self.relative_ned[:,1:2], 2) + np.power(self.relative_ned[:,0:1], 2))
        self.relative_angle = np.arctan2(self.relative_ned[:,1:2],  self.relative_ned[:,0:1]) * 180 / np.pi
        
        # convert relative angle to be in the range of -180 and 180.
        self.relative_angle = self.limit_angle(self.relative_angle)
        
        # compute the relative angle detected by camera
        self.camera_det_angle = self.limit_angle( -1 *  self.camera_c0_angle)
        
        # compute the relative x and y from camera detection
        self.camera_n =  self.camera_c0_dis * np.cos(self.camera_det_angle * np.pi / 180.0) 
        self.camera_e =  self.camera_c0_dis * np.sin(self.camera_det_angle * np.pi / 180.0) 

    def limit_angle (self, angle):
        """ this function is to find the angle value larger than 180 degree and smaller than -180 degree.
        Convert these values in the range of -180 and 180"""
        indx = np.nonzero(angle > 180.0)
        angle[indx] = angle[indx] - 360        

        indx = np.nonzero(angle < -180.0)
        angle[indx] = angle[indx] + 360
        return angle
        
    def plot_time_stamp (self):
        time_stamp = self.time
        plt.figure()
        plt.plot(np.diff(time_stamp))
        plt.ylabel('time stamp difference (second)')
        plt.show()    

    def arrange_dat (self):
        """ remove the unnessary data, and change the data format for further process"""
    
        host_data_non_zero   = np.nonzero(self.host_lat)
        target_data_non_zero = np.nonzero(self.target_lat)
        
        # find the non zero data from the host and target data
        start_indx = max(host_data_non_zero[0][0], target_data_non_zero[0][0])
        end_indx   = min(host_data_non_zero[0][-1], target_data_non_zero[0][-1])
        
        # choose the proper host vehicle data range
        self.host_lat = np.reshape(self.host_lat[start_indx:end_indx], (end_indx - start_indx, 1)) 
        self.host_lon = np.reshape(self.host_lon[start_indx:end_indx], (end_indx - start_indx, 1)) 
        self.host_ele = np.reshape(self.host_ele[start_indx:end_indx], (end_indx - start_indx, 1))       
        self.host_hea = np.reshape(self.host_hea[start_indx:end_indx], (end_indx - start_indx, 1)) 
        self.host_speed  = np.reshape(self.host_speed[start_indx:end_indx], (end_indx - start_indx, 1)) 
        
        # convert the host vehicle to the correct format
        self.host_lat = self.convert_to_rad(self.host_lat)
        self.host_lon = self.convert_to_rad(self.host_lon)
        self.host_ele = self.host_ele.astype(float)/10.0
        
        # choose the proper target vehicle data range
        self.target_lat = np.reshape(self.target_lat[start_indx:end_indx], (end_indx - start_indx, 1)) 
        self.target_lon = np.reshape(self.target_lon[start_indx:end_indx], (end_indx - start_indx, 1)) 
        self.target_ele = np.reshape(self.target_ele[start_indx:end_indx], (end_indx - start_indx, 1))       
        self.target_hea = np.reshape(self.target_hea[start_indx:end_indx], (end_indx - start_indx, 1)) 
        self.target_speed = np.reshape(self.target_speed[start_indx:end_indx], (end_indx - start_indx, 1)) 
        self.v2v_new_data = np.reshape(self.v2v_new_data[start_indx:end_indx], (end_indx - start_indx, 1))
        
        self.target_lat = self.convert_to_rad(self.target_lat)
        self.target_lon = self.convert_to_rad(self.target_lon)
        self.target_ele = self.target_ele.astype(float)/10.0
        
        # host vehicle heading data
        self.host_hea = self.host_hea.astype(float) * 0.0125
        self.host_hea = self.limit_angle(self.host_hea)
        
        # host vehicle speed data process
        self.host_speed  = self.host_speed.astype(float) * 0.02

         # target vehicle heading data
        self.target_hea = self.target_hea.astype(float) * 0.0125
        self.target_hea = self.limit_angle(self.target_hea)
        
        # host vehicle speed data process
        self.target_speed  = self.target_speed.astype(float) * 0.02
                
        # compute the relative speed between the host and target vehicle
        self.relative_v_n = self.target_speed * np.cos(self.target_hea * np.pi / 180.0) -                             self.host_speed * np.cos(self.host_hea * np.pi / 180.0)
        self.relative_v_e = self.target_speed * np.sin(self.target_hea * np.pi / 180.0) -                             self.host_speed * np.sin(self.host_hea * np.pi / 180.0)    
        
        self.time = np.reshape(self.time[start_indx:end_indx], (end_indx - start_indx, 1)) 
        self.time = self.time.astype(float)/1000.0  # convert the ms to second
        self.time = self.time - self.time[0]        # set time start as zero
    
        # convert camera detected offset to angle      
        self.camera_c0_angle = np.reshape(self.camera_c0_angle[start_indx:end_indx], (end_indx - start_indx, 1)) 
        self.camera_c0_dis   = np.reshape(self.camera_c0_dis[start_indx:end_indx], (end_indx - start_indx, 1)) 
        self.camera_new_data = np.reshape(self.camera_new_data[start_indx:end_indx], (end_indx - start_indx, 1))
        
        self.camera_c0_angle = self.convert_offset2angle(self.camera_c0_angle)
        
    def convert_offset2angle (self, offset):
        """ this function is to convert camera detected car position to the actual angle based on the camera parameters """
        half_hori_fov = np.pi / 6.0  # the camera horizontal FOV is 60 degree
        tan_half_fov  = np.tan(half_hori_fov)
        return (np.arctan(offset / 0.5 *tan_half_fov)) * 180 / np.pi
    
    def comp_ned(self):    
        """ this function is to get the NED value from GPS data """
        self.host_xyz = self.llh2xyz(self.host_lat, self.host_lon, self.host_ele)
        self.host_ned = self.xyz2ned (self.host_xyz[0, :],self.host_xyz)
        
        self.target_xyz = self.llh2xyz(self.target_lat, self.target_lon, self.target_ele)
        self.target_ned = self.xyz2ned (self.host_xyz[0, :],self.target_xyz)
        
        # compute the lative NED 
        self.relative_ned = self.target_ned - self.host_ned
        
    def plot_location(self, x, y, title = []):        
        """ this function is to plot the vehicle 2-D position in north and east"""
        
        # find the range of the x and y data, and add some margins
        min_val = np.minimum(x.min(), y.min()) - 1
        max_val = np.maximum(x.max(), y.max()) + 1
        
        plt.figure()
        plt.plot(x, y) # plot the x/y values
        plt.axis([min_val, max_val, min_val, max_val])
        plt.xlabel('North (meter)')
        plt.ylabel('East (meter)')
        plt.title(title)
        plt.show()
        
    def convert_to_rad (self, value):
        return value.astype(float) * np.pi / 180.0 / np.power(10, 7)
                
    def llh2xyz(self,lat, lon, height):
        """ this function is to convert latitude, longtitude, and height to XYZ """        
        # earth parameters in meters
        equ_radius   = 6378137.0
        polar_radius = 6356752.0
        
        N = (equ_radius ** 2) / np.sqrt ((equ_radius ** 2) * np.power(np.cos(lat), 2) + (polar_radius ** 2) *  np.power(np.sin(lat), 2))
        x = (N + height) * np.cos(lat) * np.cos(lon)
        y = (N + height) * np.cos(lat) * np.sin(lon)
        z = (((polar_radius / equ_radius) ** 2) * N + height) * np.sin(lat)                         
        return np.concatenate((x, y, z), axis = 1)
    
    def xyz2ned (self,xyz_ref, xyz):
        """ this function is to convert XYZ coordinate to NED system """
        REF_ELLIP_AOVRB = 1.003364089820976
        REF_ELLIP_ECPSQ = 0.006739496742276435
        REF_ELLIP_ECCSQ = 0.006694379990141317
        REF_ELLIP_A     = 6378137
        REF_ELLIP_B     = 6356752.31424518 
         
        p = np.sqrt (xyz_ref[0] * xyz_ref[0] + xyz_ref[1] * xyz_ref[1]) 
        
        slon = xyz_ref[1] / p
        clon = xyz_ref[0] / p
        
        # compuate latritude 
        tu = xyz_ref[2] / p * REF_ELLIP_AOVRB
                                                  
        if (np.abs(tu) < 0.0001):
            su = tu
            cu = 1.0 - 0.5 * tu * tu
        else:
            cusq = 1.0 / (1.0 + tu * tu)
            su = np.sqrt(1.0 - cusq) 
            if (tu < 0.0):
                su = -1.0 * su
            cu = np.sqrt(cusq)

        qs = xyz_ref[2] +  (REF_ELLIP_ECPSQ * REF_ELLIP_B * su)*su*su
        qc  = p - ( REF_ELLIP_ECCSQ * REF_ELLIP_A * cu) * cu * cu
        q = np.sqrt(qs * qs + qc * qc)

        clat = qc / q
        slat = qs / q
        # compute Rt2e
        Rt2e_0 = -slat * clon
        Rt2e_1 = -slon  
        Rt2e_2 = -clat * clon 
        Rt2e_3 = -slat * slon   
        Rt2e_4 = clon
        Rt2e_5 = -clat * slon 
        Rt2e_6 = clat   
        Rt2e_7 = 0.0     
        Rt2e_8 = -slat
                                                  
        dX = xyz[:,0] - xyz_ref[0]
        dY = xyz[:,1] - xyz_ref[1]
        dZ = xyz[:,2] - xyz_ref[2]
                                                  
        n = Rt2e_0 * dX + Rt2e_3 * dY + Rt2e_6 * dZ;
        e = Rt2e_1 * dX + Rt2e_4 * dY + Rt2e_7 * dZ;
        d = Rt2e_2 * dX + Rt2e_5 * dY + Rt2e_8 * dZ;   
        
        n_shape = n.shape
        
        # reshape n, e, d for concatenate function
        n = np.reshape(n, (n_shape[0], 1))
        e = np.reshape(e, (n_shape[0], 1))
        d = np.reshape(d, (n_shape[0], 1))
        return np.concatenate((n, e, d), axis = 1)

