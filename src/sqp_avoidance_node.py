#!/usr/bin/env python3
import time
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from f110_msgs.msg import Wpnt, WpntArray, Obstacle, ObstacleArray, OTWpntArray, OpponentTrajectory, OppWpnt
from dynamic_reconfigure.msg import Config
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Float32MultiArray, Float32
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from frenet_converter.frenet_converter import FrenetConverter
from std_msgs.msg import Bool
from copy import deepcopy

class SQPAvoidanceNode:
    """
    This class implements a ROS node that creates a overtaking trajectory around osbtacles and opponents.

    It subscribes to the following topics:
        - `/perception/obstacles`: Subscribes to the obstacle array.
        - `/collision_prediction/obstacles`: Subscribes to the predicted obstacle array (ROCs).
        - `/car_state/odom_frenet`: Subscribes to the car state in Frenet coordinates.
        - `/global_waypoints`: Subscribes to the global waypoints.
        - `/global_waypoints_scaled`: Subscribes to the scaled global waypoints.
        - `/global_waypoints_updated`: Subscribes to the updated global waypoints.
        - `/local_waypoints`: Subscribes to the local waypoints.
        - `/opponent_waypoints`: Subscribes to the opponent waypoints.
        - `/ot_section_check`: Subscribes to the overtaking section check.
        - `/dynamic_sqp_tuner_node/parameter_updates`: Subscribes to the dynamic reconfigure updates.

    The node publishes the following topics:
        - `/planner/avoidance/markers_sqp`: Publishes the markers for the avoidance trajectory.
        - `/planner/avoidance/otwpnts`: Publishes the overtaking waypoints.
        - `/planner/avoidance/merger`: Publishes the merger region of the overtaking trajectory.
        - `/planner/pspliner_sqp/latency`: Publishes the latency of the SQP solver. (Only if measure is set to True)
    """

    def __init__(self):
        # Initialize node
        rospy.init_node('sqp_avoidance_node')
        self.rate = rospy.Rate(20)

        # Params
        self.frenet_state = Odometry()
        self.local_wpnts = None
        self.lookahead = 15
        self.past_avoidance_d = []
        # Scaled waypoints params
        self.scaled_wpnts = None
        self.scaled_wpnts_msg = WpntArray()
        self.scaled_vmax = None
        self.scaled_max_idx = None
        self.scaled_max_s = None
        self.scaled_delta_s = None
        # Updated waypoints params
        self.wpnts_updated = None
        self.max_s_updated = None
        self.max_idx_updated = None
        # Obstalces params
        self.obs = ObstacleArray()
        self.obs_perception = ObstacleArray()
        self.obs_predict = ObstacleArray()
        # Opponent waypoint params
        self.opponent_waypoints = OpponentTrajectory()
        self.max_opp_idx = None
        self.opponent_wpnts_sm = None
        # OT params
        self.last_ot_side = ""
        self.ot_section_check = False
        # Solver params
        self.min_radius = 0.05  # wheelbase / np.tan(max_steering)
        self.max_kappa = 1/self.min_radius
        self.width_car = 0.28
        self.avoidance_resolution = 20
        self.back_to_raceline_before = 5
        self.back_to_raceline_after = 5
        self.obs_traj_tresh = 2

        # Dynamic sovler params
        self.down_sampled_delta_s = None
        self.global_traj_kappas = None

        # ROS Parameters
        self.opponent_traj_topic = '/opponent_trajectory'
        self.measure = rospy.get_param("/measure", False)

        # Dynamic reconf params
        self.avoid_static_obs = True

        self.converter = None
        self.global_waypoints = None

        # Subscribers
        rospy.Subscriber("/perception/obstacles", ObstacleArray, self.obs_perception_cb)
        rospy.Subscriber("/collision_prediction/obstacles", ObstacleArray, self.obs_prediction_cb)
        rospy.Subscriber("/car_state/odom_frenet", Odometry, self.state_cb)
        rospy.Subscriber("/global_waypoints_scaled", WpntArray, self.scaled_wpnts_cb)
        rospy.Subscriber("/local_waypoints", WpntArray, self.local_wpnts_cb)
        rospy.Subscriber("/dynamic_sqp_tuner_node/parameter_updates", Config, self.dyn_param_cb)
        rospy.Subscriber("/global_waypoints", WpntArray, self.gb_cb)
        rospy.Subscriber("/global_waypoints_updated", WpntArray, self.updated_wpnts_cb)
        rospy.Subscriber(self.opponent_traj_topic, OpponentTrajectory, self.opponent_trajectory_cb)
        rospy.Subscriber("/ot_section_check", Bool, self.ot_sections_check_cb)
        # Publishers
        self.mrks_pub = rospy.Publisher("/planner/avoidance/markers_sqp", MarkerArray, queue_size=10)
        self.evasion_pub = rospy.Publisher("/planner/avoidance/otwpnts", OTWpntArray, queue_size=10)
        self.merger_pub = rospy.Publisher("/planner/avoidance/merger", Float32MultiArray, queue_size=10)
        if self.measure:
            self.measure_pub = rospy.Publisher("/planner/pspliner_sqp/latency", Float32, queue_size=10)

        self.converter = self.initialize_converter()

    ### Callbacks ###
    def obs_perception_cb(self, data: ObstacleArray):
        self.obs_perception = data
        self.obs_perception.obstacles = [obs for obs in data.obstacles if obs.is_static == True]
        if self.avoid_static_obs == True:
            self.obs.header = data.header
            self.obs.obstacles = self.obs_perception.obstacles + self.obs_predict.obstacles

    def obs_prediction_cb(self, data: ObstacleArray):
        self.obs_predict = data
        self.obs = self.obs_predict
        if self.avoid_static_obs == True:
            self.obs.obstacles = self.obs.obstacles + self.obs_perception.obstacles

    def state_cb(self, data: Odometry):
        self.frenet_state = data

    def gb_cb(self, data: WpntArray):
        self.global_waypoints = np.array([[wpnt.x_m, wpnt.y_m] for wpnt in data.wpnts])

    # Everything is refered to the SCALED global waypoints
    def scaled_wpnts_cb(self, data: WpntArray):
        self.scaled_wpnts = np.array([[wpnt.s_m, wpnt.d_m] for wpnt in data.wpnts])
        self.scaled_wpnts_msg = data
        v_max = np.max(np.array([wpnt.vx_mps for wpnt in data.wpnts]))
        if self.scaled_vmax != v_max:
            self.scaled_vmax = v_max
            self.scaled_max_idx = data.wpnts[-1].id
            self.scaled_max_s = data.wpnts[-1].s_m
            self.scaled_delta_s = data.wpnts[1].s_m - data.wpnts[0].s_m

    def updated_wpnts_cb(self, data: WpntArray):
        self.wpnts_updated = data.wpnts[:-1]
        self.max_s_updated = self.wpnts_updated[-1].s_m
        self.max_idx_updated = self.wpnts_updated[-1].id

    def local_wpnts_cb(self, data: WpntArray):
        self.local_wpnts = np.array([[wpnt.s_m, wpnt.d_m] for wpnt in data.wpnts])

    def opponent_trajectory_cb(self, data: OpponentTrajectory):
        self.opponent_waypoints = data.oppwpnts
        self.max_opp_idx = len(data.oppwpnts)-1
        self.opponent_wpnts_sm = np.array([wpnt.s_m for wpnt in data.oppwpnts])

    def ot_sections_check_cb(self, data: Bool):
        self.ot_section_check = data.data

    # Callback triggered by dynamic spline reconf
    def dyn_param_cb(self, params: Config):
        self.evasion_dist = rospy.get_param("dynamic_sqp_tuner_node/evasion_dist", 0.65)
        self.obs_traj_tresh = rospy.get_param("dynamic_sqp_tuner_node/obs_traj_tresh", 1.5)
        self.spline_bound_mindist = rospy.get_param("dynamic_sqp_tuner_node/spline_bound_mindist", 0.2)
        self.lookahead = rospy.get_param("dynamic_sqp_tuner_node/lookahead_dist", 15)
        self.avoidance_resolution = rospy.get_param("dynamic_sqp_tuner_node/avoidance_resolution", 20)
        self.back_to_raceline_before = rospy.get_param("dynamic_sqp_tuner_node/back_to_raceline_before", 5)
        self.back_to_raceline_after = rospy.get_param("dynamic_sqp_tuner_node/back_to_raceline_after", 5)
        self.avoid_static_obs = rospy.get_param("dynamic_sqp_tuner_node/avoid_static_obs", True)

        print(
            f"[Planner] Dynamic reconf triggered new spline params: \n"
            f" Evasion apex distance: {self.evasion_dist} [m],\n"
            f" Obstacle trajectory treshold: {self.obs_traj_tresh} [m]\n"
            f" Spline boundary mindist: {self.spline_bound_mindist} [m]\n"
            f" Lookahead distance: {self.lookahead} [m]\n"
            f" Avoid static obstacles: {self.avoid_static_obs}\n"
            f" Avoidance resolution: {self.avoidance_resolution}\n"
            f" Back to raceline before: {self.back_to_raceline_before} [m]\n"
            f" Back to raceline after: {self.back_to_raceline_after} [m]\n"
        )

    def loop(self):
        # Wait for critical Messages and services
        rospy.loginfo("[OBS Spliner] Waiting for messages and services...")
        rospy.wait_for_message("/global_waypoints_scaled", WpntArray)
        rospy.wait_for_message("/car_state/odom", Odometry)
        rospy.wait_for_message("/dynamic_sqp_tuner_node/parameter_updates", Config)
        rospy.wait_for_message("/local_waypoints",WpntArray)
        rospy.loginfo("[OBS Spliner] Ready!")

        while not rospy.is_shutdown():
            start_time = time.perf_counter()
            obs = deepcopy(self.obs)
            mrks = MarkerArray()
            frenet_state = self.frenet_state
            self.current_d = frenet_state.pose.pose.position.y
            self.cur_s = frenet_state.pose.pose.position.x
            
            # Obstacle pre-processing
            obs.obstacles = sorted(obs.obstacles, key=lambda obs: obs.s_start)
            considered_obs = []
            for obs in obs.obstacles:               
                if abs(obs.d_center) < self.obs_traj_tresh and (obs.s_start - self.cur_s) % self.scaled_max_s < self.lookahead:
                    considered_obs.append(obs)
            
            # If there is an obstacle and we are in OT section
            if len(considered_obs) > 0 and self.ot_section_check == True:             
                evasion_x, evasion_y, evasion_s, evasion_d, evasion_v = self.sqp_solver(considered_obs, frenet_state.pose.pose.position.x)
                # Publish merge reagion if evasion track has been found
                if len(evasion_s) > 0:
                    self.merger_pub.publish(Float32MultiArray(data=[considered_obs[-1].s_end%self.scaled_max_s, evasion_s[-1]%self.scaled_max_s]))

            # IF there is no point in overtaking anymore delte all markers
            else:
                mrks = MarkerArray()
                del_mrk = Marker(header=rospy.Header(stamp=rospy.Time.now()))
                del_mrk.action = Marker.DELETEALL
                mrks.markers = []
                mrks.markers.append(del_mrk)
                self.mrks_pub.publish(mrks)
        
            # publish latency
            if self.measure:
                self.measure_pub.publish(Float32(data=time.perf_counter() - start_time))
            self.rate.sleep()

    def sqp_solver(self, considered_obs: list, cur_s: float):
        danger_flag = False
        # Get the initial guess of the overtaking side (see spliner)
        initial_guess_object = self.group_objects(considered_obs)
        initial_guess_object_start_idx = np.abs(self.scaled_wpnts - initial_guess_object.s_start).argmin()
        initial_guess_object_end_idx = np.abs(self.scaled_wpnts - initial_guess_object.s_end).argmin()
        # Get array of indexes of the global waypoints overlapping with the ROC
        gb_idxs = np.array(range(initial_guess_object_start_idx, initial_guess_object_start_idx + (initial_guess_object_end_idx - initial_guess_object_start_idx)%self.scaled_max_idx))%self.scaled_max_idx
        # If the ROC is too short, we take the next 20 waypoints
        if len(gb_idxs) < 20:
            gb_idxs = [int(initial_guess_object.s_center / self.scaled_delta_s + i) % self.scaled_max_idx for i in range(20)]

        side, initial_apex = self._more_space(initial_guess_object, self.scaled_wpnts_msg.wpnts, gb_idxs)
        kappas = np.array([self.scaled_wpnts_msg.wpnts[gb_idx].kappa_radpm for gb_idx in gb_idxs])
        max_kappa = np.max(np.abs(kappas))
        outside = "left" if np.sum(kappas) < 0 else "right"

        # Enlongate the ROC if our initial guess suggests that we are overtaking on the outside
        if side == outside:
            for i in range(len(considered_obs)):
                considered_obs[i].s_end = considered_obs[i].s_end + (considered_obs[i].s_end - considered_obs[i].s_start)%self.max_s_updated * max_kappa * (self.width_car + self.evasion_dist)

        min_s_obs_start = self.scaled_max_s
        max_s_obs_end = 0
        for obs in considered_obs:
            if obs.s_start < min_s_obs_start:
                min_s_obs_start = obs.s_start
            if obs.s_end > max_s_obs_end:
                max_s_obs_end = obs.s_end
            # Check if it is a really wide obstacle
            if obs.d_left > 3 or obs.d_right < -3:
                danger_flag = True

        # Get local waypoints to check where we are and where we are heading
        # If we are closer than threshold to the opponent use the first two local waypoints as start points
        start_avoidance = max((min_s_obs_start - self.back_to_raceline_before), cur_s)
        end_avoidance = max_s_obs_end + self.back_to_raceline_after

        # Get a downsampled version for s avoidance points
        s_avoidance = np.linspace(start_avoidance, end_avoidance, self.avoidance_resolution)
        self.down_sampled_delta_s = s_avoidance[1] - s_avoidance[0]
        # Get the closest scaled waypoint for every s avoidance point (down sampled)
        scaled_wpnts_indices = np.array([np.abs(self.scaled_wpnts[:, 0] - s % self.scaled_max_s).argmin() for s in s_avoidance]) 
        # Get the scaled waypoints for every s avoidance point idx
        corresponding_scaled_wpnts = [self.scaled_wpnts_msg.wpnts[i] for i in scaled_wpnts_indices]
        # Get the boundaries for every s avoidance point
        bounds = np.array([(-wpnt.d_right + self.spline_bound_mindist, wpnt.d_left - self.spline_bound_mindist) for wpnt in corresponding_scaled_wpnts])

        # Calculate curvature at each point using numerical differentiation
        # k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        x_global_points = np.array([wpnt.x_m for wpnt in corresponding_scaled_wpnts])
        y_global_points = np.array([wpnt.y_m for wpnt in corresponding_scaled_wpnts])
        x_prime = np.diff(x_global_points)
        x_prime = np.where(x_prime == 0, 1e-6, x_prime) # Avoid division by zero
        y_prime = np.diff(y_global_points)
        y_prime = np.where(y_prime == 0, 1e-6, y_prime) # Avoid division by zero
        x_prime_prime = np.diff(x_prime)
        y_prime_prime = np.diff(y_prime)
        x_prime = x_prime[:-1] # Make it the same length as x_prime_prime
        y_prime = y_prime[:-1] # Make it the same length as y_prime_prime
        self.global_traj_kappas = (x_prime*y_prime_prime - y_prime*x_prime_prime) / ((x_prime**2 + y_prime**2)**(3/2))
       
        # Create a list of indices which overlap with the obstacles
        # Get the centerline of the obstacles and enforce a min distance to the obstacles
        self.obs_downsampled_indices = np.array([])
        self.obs_downsampled_center_d = np.array([])
        self.obs_downsampled_min_dist = np.array([])

        for obs in considered_obs:
            obs_idx_start = np.abs(s_avoidance - obs.s_start).argmin()
            obs_idx_end = np.abs(s_avoidance - obs.s_end).argmin()

            if obs_idx_start < len(s_avoidance) - 2: # Sanity check
                if obs.is_static == True or obs_idx_end == obs_idx_start:
                    if obs_idx_end == obs_idx_start:
                        obs_idx_end = obs_idx_start + 1
                    self.obs_downsampled_indices = np.append(self.obs_downsampled_indices, np.arange(obs_idx_start, obs_idx_end + 1))
                    self.obs_downsampled_center_d = np.append(self.obs_downsampled_center_d, np.full(obs_idx_end - obs_idx_start + 1, (obs.d_left + obs.d_right) / 2))
                    self.obs_downsampled_min_dist = np.append(self.obs_downsampled_min_dist, np.full(obs_idx_end - obs_idx_start + 1, (obs.d_left - obs.d_right) / 2 + self.width_car + self.evasion_dist))
                else:
                    indices = np.arange(obs_idx_start, obs_idx_end + 1)
                    self.obs_downsampled_indices = np.append(self.obs_downsampled_indices, indices)
                    opp_wpnts_idx = [np.abs(self.opponent_wpnts_sm - s_avoidance[int(idx)]%self.max_opp_idx).argmin() for idx in indices]
                    d_opp_downsampled_array = np.array([self.opponent_waypoints[opp_idx].d_m for opp_idx in opp_wpnts_idx])                    
                    self.obs_downsampled_center_d = np.append(self.obs_downsampled_center_d, d_opp_downsampled_array)
                    self.obs_downsampled_min_dist = np.append(self.obs_downsampled_min_dist, np.full(obs_idx_end - obs_idx_start + 1, self.width_car + self.evasion_dist))
            else:
                rospy.loginfo("[OBS Spliner] Obstacle end index is smaller than start index")
                rospy.loginfo("[OBS Spliner] len obs: " + str(len(considered_obs)) + "obs_start:" + str(obs.s_start) + "obs_end:" + str(obs.s_end) + " obs_idx_start: " + str(obs_idx_start) + " obs_idx_end: " + str(obs_idx_end) + " len s_avoidance: " + str(len(s_avoidance)) + "s avoidance 0:" + str(s_avoidance[0]) + " s avoidance -1: " + str(s_avoidance[-1]))    

    
        self.obs_downsampled_indices = self.obs_downsampled_indices.astype(int)

        # Get the min radius
        # Clip speed between 1 and 7 m/s
        clipped_speed = np.clip(self.frenet_state.twist.twist.linear.x, 1, 6.5)
        # Get the minimum of clipped speed and the updated speed of the first waypoints
        radius_speed = min([clipped_speed, self.wpnts_updated[(scaled_wpnts_indices[0])%self.max_idx_updated].vx_mps])
        # Interpolate the min_radius with speeds between 0.2 and 7 m
        self.min_radius = np.interp(radius_speed, [1, 6, 7], [0.2, 2, 4])
        self.max_kappa = 1/self.min_radius

        if len(self.past_avoidance_d) == 0:
            initial_guess = np.full(len(s_avoidance), initial_apex)

        elif len(self.past_avoidance_d) > 0:
            initial_guess = self.past_avoidance_d
        else:
            #TODO: Remove -> print("this happend")
            if self.last_ot_side == "left":
                initial_guess = np.full(len(s_avoidance), 2)
            else:
                initial_guess = np.full(len(s_avoidance), -2)
            
        result = self.solve_sqp(initial_guess, bounds)
    
        # if len(self.obs_downsampled_indices) < 2 or danger_flag == True:
        #     result.success = False

        if result.success == True:
            # Create a new s array for the global waypoints as close to delta s as possible
            n_global_avoidance_points = int((end_avoidance - start_avoidance) / self.scaled_delta_s)
            s_array = np.linspace(start_avoidance, end_avoidance, n_global_avoidance_points)
            # Interpolate corresponding d values
            evasion_d = np.interp(s_array, s_avoidance, result.x)
            # Solve rap around problem
            evasion_s = np.mod(s_array, self.scaled_max_s)
            # Get the corresponding x and y values
            resp = self.converter.get_cartesian(evasion_s, evasion_d)
            evasion_x = resp[0, :]
            evasion_y = resp[1, :]
            # Get the corresponding v values
            downsampled_v = np.array([wpnt.vx_mps for wpnt in corresponding_scaled_wpnts])
            evasion_v = np.interp(s_array, s_avoidance, downsampled_v)
            # Create a new evasion waypoint message
            evasion_wpnts_msg = OTWpntArray(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"))
            evasion_wpnts = []
            evasion_wpnts = [Wpnt(id=len(evasion_wpnts), s_m=s, d_m=d, x_m=x, y_m=y, vx_mps= v) for x, y, s, d, v in zip(evasion_x, evasion_y, evasion_s, evasion_d, evasion_v)]
            evasion_wpnts_msg.wpnts = evasion_wpnts
            self.past_avoidance_d = result.x
            mean_d = np.mean(evasion_d)
            if mean_d > 0:
                self.last_ot_side = "left"
            else:
                self.last_ot_side = "right"
            # print("[OBS Spliner] SQP solver successfull")

        else:
            evasion_x = []
            evasion_y = []
            evasion_s = []
            evasion_d = []
            evasion_v = []
            evasion_wpnts_msg = OTWpntArray(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"))
            evasion_wpnts_msg.wpnts = []
            self.past_avoidance_d = []
        
        self.evasion_pub.publish(evasion_wpnts_msg)
        self.visualize_sqp(evasion_s, evasion_d, evasion_x, evasion_y, evasion_v) 

        return evasion_x, evasion_y, evasion_s, evasion_d, evasion_v


    ### Optimal Trajectory Solver ###
    def objective_function(self, d):
        return np.sum((d) ** 2) * 10  + np.sum(np.diff(np.diff(d))**2) * 100 + (np.diff(d)[0] ** 2) * 1000

    ## Constraint functions ##
    def start_on_raceline_constraint(self, d): # And end on raceline
        return np.array([0.02 - abs(d[0] - self.current_d), 0.02 - abs(d[-2]), 0.02 - abs(d[-1])])        

    def obstacle_constraint(self, d):
        distance_to_obstacle = np.abs(d[self.obs_downsampled_indices] - self.obs_downsampled_center_d)
        violation = distance_to_obstacle - self.obs_downsampled_min_dist
        return violation
    
    # Prevents points from jumping trhough obstacles due to resoultion isses
    def consecutive_points_constraint(self, d):
        # Extract the relevant points
        points = d[self.obs_downsampled_indices]
    
        # Check the condition for each pair of consecutive points
        violations = []
        for i in range(len(points) - 1):
            if not ((points[i] > self.obs_downsampled_center_d[i] and points[i+1] > self.obs_downsampled_center_d[i+1]) or
                    (points[i] < self.obs_downsampled_center_d[i] and points[i+1] < self.obs_downsampled_center_d[i+1])):
                violations.append(-1)  # Add a violation as a negative value if the condition is not met
            else:
                violations.append(1)
        return violations

    def turning_radius_constraint(self, d):
        # Calculate curvature at each point using numerical differentiation
        # k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        # x' = self.down_sampled_delta_s, x'' = 0
        
        y_prime = np.diff(d)
        y_prime = np.where(y_prime == 0, 1e-6, y_prime) # Avoid division by zero
        y_prime_prime = np.diff(y_prime)
        y_prime = y_prime[:-1] # Make it the same length as y_prime_prime
        
        kappa = (self.down_sampled_delta_s * y_prime_prime) / ((self.down_sampled_delta_s ** 2) ** (3/2))
        # np.diff losses last two points so we delete them from self.global_traj_kappas
        total_kappa = self.global_traj_kappas - kappa
        violation = self.max_kappa - abs(total_kappa)
        return violation
    
    # The arctan of of (d[1]-d[0])/ delta_s_sample_points < than 45 degrees
    def first_point_constraint(self, d):
        return np.array([self.down_sampled_delta_s - abs(d[1]-d[0])])

    def combined_equality_constraints(self, d):
        return self.start_on_raceline_constraint(d)

    def combined_inequality_constraints(self, d):
        return np.concatenate([self.obstacle_constraint(d), self.consecutive_points_constraint(d), self.turning_radius_constraint(d), self.first_point_constraint(d)]) #

    def solve_sqp(self, d_array, track_boundaries):
        result = minimize(
        self.objective_function, d_array, method='SLSQP', jac='10-point',
        bounds=track_boundaries,
        constraints=[
            {'type': 'eq', 'fun': self.combined_equality_constraints},
            {'type': 'ineq', 'fun': self.combined_inequality_constraints}
            ],
        options={'ftol': 1e-1, 'maxiter': 20, 'disp': False},
        )
        return result

    def group_objects(self, obstacles: list):
        # Group obstacles that are close to each other
        initial_guess_object = obstacles[0]
        for obs in obstacles:
            if obs.d_left > initial_guess_object.d_left:
                initial_guess_object.d_left = obs.d_left
            if obs.d_right < initial_guess_object.d_right:
                initial_guess_object.d_right = obs.d_right
            if obs.s_start < initial_guess_object.s_start:
                initial_guess_object.s_start = obs.s_start
            if obs.s_end > initial_guess_object.s_end:
                initial_guess_object.s_end = obs.s_end
        initial_guess_object.s_center = (initial_guess_object.s_start + initial_guess_object.s_end) / 2
        return initial_guess_object

    def _more_space(self, obstacle: Obstacle, gb_wpnts, gb_idxs):
        left_boundary_mean = np.mean([gb_wpnts[gb_idx].d_left for gb_idx in gb_idxs])
        right_boundary_mean = np.mean([gb_wpnts[gb_idx].d_right for gb_idx in gb_idxs])
        left_gap = abs(left_boundary_mean - obstacle.d_left)
        right_gap = abs(right_boundary_mean + obstacle.d_right)
        min_space = self.evasion_dist + self.spline_bound_mindist

        if right_gap > min_space and left_gap < min_space:
            # Compute apex distance to the right of the opponent
            d_apex_right = obstacle.d_right - self.evasion_dist
            # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
            if d_apex_right > 0:
                d_apex_right = 0
            return "right", d_apex_right

        elif left_gap > min_space and right_gap < min_space:
            # Compute apex distance to the left of the opponent
            d_apex_left = obstacle.d_left + self.evasion_dist
            # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
            if d_apex_left < 0:
                d_apex_left = 0
            return "left", d_apex_left
        else:
            candidate_d_apex_left = obstacle.d_left + self.evasion_dist
            candidate_d_apex_right = obstacle.d_right - self.evasion_dist

            if abs(candidate_d_apex_left) <= abs(candidate_d_apex_right):
                # If we overtake to the left of the opponent BUT the apex is to the right of the raceline, then we set the apex to 0
                if candidate_d_apex_left < 0:
                    candidate_d_apex_left = 0
                return "left", candidate_d_apex_left
            else:
                # If we overtake to the right of the opponent BUT the apex is to the left of the raceline, then we set the apex to 0
                if candidate_d_apex_right > 0:
                    candidate_d_apex_right = 0
                return "right", candidate_d_apex_right
    
    ### Visualize SQP Rviz###
    def visualize_sqp(self, evasion_s, evasion_d, evasion_x, evasion_y, evasion_v):
        mrks = MarkerArray()
        if len(evasion_s) == 0:
            pass
        else:
            resp = self.converter.get_cartesian(evasion_s, evasion_d)
            for i in range(len(evasion_s)):
                mrk = Marker(header=rospy.Header(stamp=rospy.Time.now(), frame_id="map"))
                mrk.type = mrk.CYLINDER
                mrk.scale.x = 0.1
                mrk.scale.y = 0.1
                mrk.scale.z = evasion_v[i] / self.scaled_vmax
                mrk.color.a = 1.0
                mrk.color.g = 0.13
                mrk.color.r = 0.63
                mrk.color.b = 0.94

                mrk.id = i
                mrk.pose.position.x = evasion_x[i]
                mrk.pose.position.y = evasion_y[i]
                mrk.pose.position.z = evasion_v[i] / self.scaled_vmax / 2
                mrk.pose.orientation.w = 1
                mrks.markers.append(mrk)
            self.mrks_pub.publish(mrks)

    def initialize_converter(self) -> bool:
            """
            Initialize the FrenetConverter object"""
            rospy.wait_for_message("/global_waypoints", WpntArray)

            # Initialize the FrenetConverter object
            converter = FrenetConverter(self.global_waypoints[:, 0], self.global_waypoints[:, 1])
            rospy.loginfo("[Spliner] initialized FrenetConverter object")

            return converter


if __name__ == "__main__":
    SQPAvoidance = SQPAvoidanceNode()
    SQPAvoidance.loop()