#!/usr/bin/env python3

import numpy as np
import rospy
from f110_msgs.msg import (ObstacleArray, OpponentTrajectory, Obstacle, Wpnt, OppWpnt, WpntArray, ProjOppTraj, ProjOppPoint)
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.interpolate import UnivariateSpline
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
import time
from collections import defaultdict
from sklearn.utils import check_random_state, check_array
from scipy.optimize import fmin_cg
from scipy.optimize import fmin_l_bfgs_b


from frenet_converter.frenet_converter import FrenetConverter

class GaussianProcessOppTraj(object):
    def __init__(self):
        #Node
        rospy.init_node('gaussian_process_opp_traj', anonymous=True)
        self.rate = rospy.Rate(10)  
        self.opp_traj = OpponentTrajectory()
        self.opp_traj_gp = OpponentTrajectory()
        self.opp_positions_in_map_frenet = [] #testing with Rosbag



        #Subscribers
        rospy.Subscriber('/proj_opponent_trajectory', ProjOppTraj, self.proj_opp_traj_cb)
        rospy.Subscriber('/global_waypoints', WpntArray, self.glb_wpnts_cb) # global waypoints

        # rospy.Subscriber('/opp/car_state/odom_frenet', Odometry, self.opp_car_state_frenet_cb) #testing with Rosbag
        rospy.Subscriber('/opponent_trajectory', OpponentTrajectory, self.opp_traj_cb) #testing with Rosbag

        #Publishers
        self.opp_traj_gp_pub = rospy.Publisher('/opponent_trajectory', OpponentTrajectory, queue_size=10)
        self.opp_traj_marker_pub = rospy.Publisher('/opponent_traj_markerarray', MarkerArray, queue_size=10)

        #Frenet Converter
        rospy.wait_for_message("/global_waypoints", WpntArray)
        self.converter = self.initialize_converter()




    #Callback
    def proj_opp_traj_cb(self,data: ProjOppTraj):
        self.proj_opp_traj = data

    def glb_wpnts_cb(self, data):
        self.waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])
        self.glb_wpnts = data
        self.track_length = data.wpnts[-1].s_m

    def opp_traj_cb(self, data: OpponentTrajectory): 
        self.opponent_trajectory = data


    #Functions
    def initialize_converter(self) -> bool:
        
        """Initialize the FrenetConverter object"""

        rospy.wait_for_message("/global_waypoints", WpntArray)

        # Initialize the FrenetConverter object
        converter = FrenetConverter(self.waypoints[:, 0], self.waypoints[:, 1])
        rospy.loginfo("[Tracking] initialized FrenetConverter object")

        return converter   


        
    #Main Loop
    def get_gp_opp_traj(self):
        # Define the constant kernels with a lower bound for the constant_value parameter
        constant_kernel1_d = ConstantKernel(constant_value=0.5, constant_value_bounds=(1e-6, 1e3))
        constant_kernel2_d = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-6, 1e3))

        # Define the Gaussian Process kernel
        
        self.kernel_d = constant_kernel1_d * Matern(length_scale=1.0, nu=3/2) + constant_kernel2_d * WhiteKernel(noise_level=1)

        self.global_wpnts = rospy.wait_for_message("/global_waypoints", WpntArray)
        self.max_velocity = max([wnpt.vx_mps for wnpt in self.global_wpnts.wpnts])
        ego_s_original = [wnpt.s_m for wnpt in self.global_wpnts.wpnts]
        ego_s_original.pop()#pop last point of original ego_s since it is double
        ego_s_doublelap = ego_s_original.copy()
        for i in range(len(ego_s_original)):
            ego_s_doublelap.append(ego_s_original[i]+self.track_length)
        self.opponent_trajectory = rospy.wait_for_message("/opponent_trajectory", OpponentTrajectory)
        proj_opp_traj = ProjOppTraj()

        while not rospy.is_shutdown():
            proj_opp_traj = rospy.wait_for_message('/proj_opponent_trajectory', ProjOppTraj)
            oppwpnts_list = self.opponent_trajectory.oppwpnts
            self.lap_count = proj_opp_traj.lapcount
            opp_on_traj = proj_opp_traj.opp_is_on_trajectory
            nr_of_points = proj_opp_traj.nrofpoints
            if opp_on_traj == False and len(proj_opp_traj.detections) != 0:
                oppwpnts_list , opp_traj_marker_array = self.predict_opponent_trajectory(ego_s_doublelap=ego_s_doublelap, proj_opp_detections=proj_opp_traj.detections, oppwpnts_list=oppwpnts_list)
                opp_traj_gp_msg = self.make_opponent_trajectory_msg(oppwpnts_list=oppwpnts_list, lap_count=self.lap_count, raw_oppenent_traj_msg=proj_opp_traj)
                #Publish
                self.opp_traj_gp_pub.publish(opp_traj_gp_msg)
                self.opp_traj_marker_pub.publish(opp_traj_marker_array)

            


    #Helper Functions
    
    def predict_opponent_trajectory(self, ego_s_doublelap: list, proj_opp_detections:list, oppwpnts_list:list):

        """Make a prediction of how the opponent will return to his trajectory"""
        #expand oppwpnts_list to a double lap
        oppwpnts_list_doublelap = oppwpnts_list.copy()
        
        for i in range(len(oppwpnts_list)):
            new_oppwpnts = OppWpnt()
            new_oppwpnts.x_m = oppwpnts_list[i].x_m
            new_oppwpnts.y_m = oppwpnts_list[i].y_m
            new_oppwpnts.s_m = oppwpnts_list[i].s_m+self.track_length
            new_oppwpnts.d_m = oppwpnts_list[i].d_m
            new_oppwpnts.proj_vs_mps = oppwpnts_list[i].proj_vs_mps
            new_oppwpnts.vd_mps = oppwpnts_list[i].vd_mps
            oppwpnts_list_doublelap.append(new_oppwpnts)
        around_origin = False
        for i in range(len(proj_opp_detections)-1):
            if  abs(proj_opp_detections[i].s - proj_opp_detections[i+1].s) > self.track_length/2 or around_origin:
                proj_opp_detections[i+1].s = proj_opp_detections[i+1].s+self.track_length
                around_origin = True

        nr_of_points_to_merge = 15
        gap = 7
        nr_of_detections = len(proj_opp_detections)
        last_s_of_detections = proj_opp_detections[-1].s

        first_index = nr_of_detections-min(nr_of_points_to_merge,nr_of_detections)

        opponent_s_sorted= [proj_opp_detections[i].s for i in range(first_index,nr_of_detections)]
        opponent_d_sorted= [proj_opp_detections[i].d for i in range(first_index,nr_of_detections)]
        opponent_vs_sorted= [proj_opp_detections[i].vs for i in range(first_index,nr_of_detections)]
        opponent_vd_sorted= [proj_opp_detections[i].vd for i in range(first_index,nr_of_detections)]
        counter = 0
        for i in range(len(oppwpnts_list_doublelap)):
            if oppwpnts_list_doublelap[i].s_m > last_s_of_detections+gap and counter < nr_of_points_to_merge:
                opponent_s_sorted.append(oppwpnts_list_doublelap[i].s_m)
                opponent_d_sorted.append(oppwpnts_list_doublelap[i].d_m)
                opponent_vs_sorted.append(oppwpnts_list_doublelap[i].proj_vs_mps)
                opponent_vd_sorted.append(oppwpnts_list_doublelap[i].vd_mps)
                counter = counter+1
        last_s_prediction = opponent_s_sorted[-1]
        first_s_prediction = opponent_s_sorted[0]

        opponent_s_sorted_reshape = np.array(opponent_s_sorted).reshape(-1, 1)
        opponent_d_sorted_reshape = np.array(opponent_d_sorted).reshape(-1, 1)
        opponent_vs_sorted_reshape = np.array(opponent_vs_sorted).reshape(-1, 1)
        opponent_vd_sorted_reshape = np.array(opponent_vd_sorted).reshape(-1, 1)

        s_pred = []
        for i in range (len(ego_s_doublelap)):
            if ego_s_doublelap[i] <= last_s_prediction and ego_s_doublelap[i] >= first_s_prediction:
                s_pred.append(ego_s_doublelap[i])
                

        def optimizer_wrapper(obj_func, initial_theta, bounds):
            solution, function_value, _ = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
            return solution, function_value
        
        gpr_d = GaussianProcessRegressor(kernel=self.kernel_d, optimizer=optimizer_wrapper)
        

        # Fit the GPR model to the data
        gpr_d.fit(opponent_s_sorted_reshape, opponent_d_sorted_reshape)
       
        # Define a range of s values for prediction
        s_pred_reshape = np.array(s_pred).reshape(-1, 1)

        # Make predictions using the GPR model
        d_pred, sigma_d = gpr_d.predict(s_pred_reshape, return_std=True)
        
        resampled_opponent_d = d_pred
        #resampled_opponent_vs = np.interp(s_pred, opponent_s_sorted, opponent_vs_sorted)
        resampled_opponent_vd = np.interp(s_pred, opponent_s_sorted, opponent_vd_sorted)

        ego_s=[s_pred[i]%self.track_length for i in range(len(s_pred))]
    
        
        resampled_wpnts_xy = self.converter.get_cartesian(ego_s , resampled_opponent_d.tolist())
        
        # replace all the entries where i have a corresponding ego_s with the predicted values
        i=0

        for i in range(len(oppwpnts_list)):
                for j in range(len(ego_s)):
                    if abs(ego_s[j]-oppwpnts_list[i].s_m) < 1e-8:
                        oppwpnts_list[i].x_m = resampled_wpnts_xy[0][j]
                        oppwpnts_list[i].y_m = resampled_wpnts_xy[1][j]
                        oppwpnts_list[i].d_m = resampled_opponent_d[j]
                        #oppwpnts_list[i].proj_vs_mps = resampled_opponent_vs[j] 
                        oppwpnts_list[i].vd_mps = resampled_opponent_vd[j]
                        oppwpnts_list[i].d_var = sigma_d[j]
                        oppwpnts_list[i].vs_var = 69
        opp_traj_marker_array = self._visualize_opponent_wpnts(oppwpnts_list=oppwpnts_list)
        return oppwpnts_list , opp_traj_marker_array


   
    

    def make_opponent_trajectory_msg(self, oppwpnts_list: list, lap_count: int, raw_oppenent_traj_msg: ObstacleArray):
         
        """Make the opponent trajectory message and return it"""

        opponent_trajectory_msg = OpponentTrajectory()
        opponent_trajectory_msg.header.seq = lap_count
        opponent_trajectory_msg.header.stamp = rospy.Time.now()
        opponent_trajectory_msg.header.frame_id = "opponent_trajectory"
        opponent_trajectory_msg.lap_count = lap_count
        opponent_trajectory_msg.oppwpnts = oppwpnts_list

        return opponent_trajectory_msg

    
    
    def _visualize_opponent_wpnts(self, oppwpnts_list: list):

        """Visualize the resampled opponent trajectory as a marker array"""
        opp_traj_marker_array = MarkerArray()
    
        i=0
        for i in range(len(oppwpnts_list)):
            marker_height = oppwpnts_list[i].proj_vs_mps/self.max_velocity

            marker = Marker(header=rospy.Header(frame_id="map"), id = i, type = Marker.CYLINDER)
            marker.pose.position.x = oppwpnts_list[i].x_m
            marker.pose.position.y = oppwpnts_list[i].y_m
            marker.pose.position.z = marker_height/2
            marker.pose.orientation.w = 1.0
            marker.scale.x = min(max(5 * oppwpnts_list[i].d_var, 0.07),0.7)
            marker.scale.y = min(max(5 * oppwpnts_list[i].d_var, 0.07),0.7)
            marker.scale.z = marker_height
            if oppwpnts_list[i].vs_var == 69:
                marker.color.a = 0.8
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            opp_traj_marker_array.markers.append(marker)#frenpy 
    

        return opp_traj_marker_array





if __name__ == '__main__':


    node = GaussianProcessOppTraj()
    node.get_gp_opp_traj()
    rospy.spin()