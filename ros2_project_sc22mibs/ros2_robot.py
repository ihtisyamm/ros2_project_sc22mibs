import threading
import sys, time
import cv2
import numpy as np
import rclpy
import random
import math
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, PoseStamped
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal


class Nav2ColorDetectionNode(Node):
    def __init__(self):
        super().__init__('nav2_color_detection')
        
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info('Waiting for Nav2 action server...')
        self.nav_client.wait_for_server()
        self.get_logger().info('Nav2 action server is available!')
        
        # direct motion control publisher 
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # navigation state
        self.navigating = False
        self.target_reached = False
        self.blue_approach_mode = False
        self.exploration_points = []
        self.current_point_index = 0
        
        # to track if we've completed the task
        self.task_completed = False
        
        # color detection flags
        self.red_found = False
        self.green_found = False
        self.blue_found = False
        self.at_optimal_distance = False
        
        # blue object tracking
        self.blue_contour = None
        self.blue_area = 0
        self.blue_center_x = 0
        self.blue_center_y = 0
        self.frame_width = 640  # Default, will be updated
        
        # area thresholds
        self.min_contour_area = 100  # Minimum area to be considered a detection
        self.blue_optimal_area = 15000  # Target area for blue object (approx. 1m distance)
        self.area_tolerance = 3000  # Tolerance for the optimal area
        
        # sensitivity for color detection
        self.sensitivity = 10
        
        # initialise cvbridge and set up a subscriber
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        self.stop_twist = Twist()
        self.forward_twist = Twist()
        self.forward_twist.linear.x = 0.2
        self.backward_twist = Twist()
        self.backward_twist.linear.x = -0.2
        
        self.timer = self.create_timer(1.0, self.navigation_timer_callback)
        
        self.generate_exploration_points()
        
        self.get_logger().info('Nav2 Color Detection Node initialized')
    
    def generate_exploration_points(self):
        # define a grid of exploration points covering the map
        grid_size = 3
        map_min_x, map_max_x = -10.0, 10.0  # Map boundaries
        map_min_y, map_max_y = -10.0, 10.0
        
        # calculate grid cell size
        cell_width = (map_max_x - map_min_x) / grid_size
        cell_height = (map_max_y - map_min_y) / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = map_min_x + (j + 0.5) * cell_width
                y = map_min_y + (i + 0.5) * cell_height
                
                # add this point with 4 different orientations (0, 90, 180, 270 degrees)
                angles = [0, math.pi/2, math.pi, 3*math.pi/2]
                for angle in angles:
                    self.exploration_points.append((x, y, angle))
        
        # add starting point
        self.exploration_points.insert(0, (0.0, 0.0, 0.0))
        
        self.get_logger().info(f'Generated {len(self.exploration_points)} exploration points')
    
    def image_callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        
        self.frame_width = image.shape[1]
        
        # Convert the RGB image into an HSV image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Set the upper and lower bounds for colors
        # Green color in HSV
        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        
        # Red color in HSV (two ranges due to the wraparound in HSV)
        hsv_red_lower1 = np.array([0, 100, 100])
        hsv_red_upper1 = np.array([10, 255, 255])
        hsv_red_lower2 = np.array([170, 100, 100])
        hsv_red_upper2 = np.array([179, 255, 255])
        
        # Blue color in HSV
        hsv_blue_lower = np.array([110 - self.sensitivity, 100, 100])
        hsv_blue_upper = np.array([110 + self.sensitivity, 255, 255])
        
        # Create masks for each color
        green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
        
        # Create red mask
        red_mask1 = cv2.inRange(hsv_image, hsv_red_lower1, hsv_red_upper1)
        red_mask2 = cv2.inRange(hsv_image, hsv_red_lower2, hsv_red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Create blue mask
        blue_mask = cv2.inRange(hsv_image, hsv_blue_lower, hsv_blue_upper)
        
        # Combine all masks for visualization
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, green_mask), blue_mask)
        
        # Create a copy of the original image
        result_image = image.copy()
        
        # Reset detection flags
        self.red_found = False
        self.green_found = False
        self.blue_found = False
        old_at_optimal_distance = self.at_optimal_distance
        self.at_optimal_distance = False
        
        # find blue contours
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(blue_contours) > 0:
            self.blue_contour = max(blue_contours, key=cv2.contourArea)
            self.blue_area = cv2.contourArea(self.blue_contour)
            
            if self.blue_area > self.min_contour_area:
                self.blue_found = True
                M = cv2.moments(self.blue_contour)
                if M["m00"] != 0:
                    self.blue_center_x = int(M["m10"] / M["m00"])
                    self.blue_center_y = int(M["m01"] / M["m00"])
                
                # draw circle around the blue object
                (x, y), radius = cv2.minEnclosingCircle(self.blue_contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(result_image, center, radius, (255, 0, 0), 3)
                
                # draw the center point
                cv2.circle(result_image, (self.blue_center_x, self.blue_center_y), 5, (255, 255, 0), -1)
                
                # add a reference line from center of image to center of object
                image_center_x = self.frame_width // 2
                cv2.line(result_image, (image_center_x, self.blue_center_y), (self.blue_center_x, self.blue_center_y), (255, 255, 0), 2)
                
                # display blue area information
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(result_image, f"BLUE AREA: {int(self.blue_area)}", (10, 30), font, 0.7, (255, 0, 0), 2)
                
                if abs(self.blue_area - self.blue_optimal_area) < self.area_tolerance:
                    cv2.putText(result_image, "STATUS: OPTIMAL DISTANCE - STOPPING", (10, result_image.shape[0] - 30), font, 0.7, (0, 255, 0), 2)
                    # add a visual indicator for optimal distance
                    cv2.rectangle(result_image, (0, 0), (result_image.shape[1]-1, result_image.shape[0]-1), 
                                    (0, 255, 0), 5)
                    self.at_optimal_distance = True
                    
                    # if this is a transition to optimal distance, log it clearly
                    if not old_at_optimal_distance:
                        self.get_logger().info("REACHED OPTIMAL DISTANCE FROM BLUE BOX - STOPPING *****")
                        # ensure we stop by publishing an immediate stop command
                        self.cmd_vel_pub.publish(self.stop_twist)
                        
                elif self.blue_area > self.blue_optimal_area:
                    cv2.putText(result_image, "STATUS: TOO CLOSE TO BLUE", 
                                (10, result_image.shape[0] - 30), font, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(result_image, "STATUS: APPROACHING BLUE", 
                                (10, result_image.shape[0] - 30), font, 0.7, (0, 255, 255), 2)
        else:
            self.blue_found = False
            self.at_optimal_distance = False
        
        # only process red and green if blue is not found
        if not self.blue_found:
            # find and process red contours
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(red_contours) > 0:
                c_red = max(red_contours, key=cv2.contourArea)
                red_area = cv2.contourArea(c_red)
                
                if red_area > self.min_contour_area:
                    self.red_found = True
                    
                    # Draw circle around the red object
                    (x, y), radius = cv2.minEnclosingCircle(c_red)
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(result_image, center, radius, (0, 0, 255), 2)
            
            # find and process green contours
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(green_contours) > 0:
                c_green = max(green_contours, key=cv2.contourArea)
                green_area = cv2.contourArea(c_green)
                
                if green_area > self.min_contour_area:
                    self.green_found = True
                    
                    # Draw circle around the green object
                    (x, y), radius = cv2.minEnclosingCircle(c_green)
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(result_image, center, radius, (0, 255, 0), 2)
        
        # if no blue detected, display appropriate status
        if not self.blue_found:
            font = cv2.FONT_HERSHEY_SIMPLEX
            if self.red_found or self.green_found:
                cv2.putText(result_image, "STATUS: SEARCHING FOR BLUE", 
                            (10, result_image.shape[0] - 30), font, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(result_image, "STATUS: EXPLORING", 
                            (10, result_image.shape[0] - 30), font, 0.7, (255, 255, 255), 2)
        
        # show color detection information
        info_bar = np.zeros((60, result_image.shape[1], 3), dtype=np.uint8)
        cv2.putText(info_bar, "RED: " + ("DETECTED" if self.red_found else "---"), 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (0, 0, 255) if self.red_found else (200, 200, 200), 2)
        cv2.putText(info_bar, "GREEN: " + ("DETECTED" if self.green_found else "---"), 
                    (250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (0, 255, 0) if self.green_found else (200, 200, 200), 2)
        cv2.putText(info_bar, "BLUE: " + ("DETECTED" if self.blue_found else "---"), 
                    (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (255, 0, 0) if self.blue_found else (200, 200, 200), 2)
        
        # Add nav mode info
        nav_mode = "BLUE APPROACH" if self.blue_approach_mode else "NAV2 EXPLORATION"
        cv2.putText(info_bar, f"MODE: {nav_mode}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Combine the info bar with the result image
        combined_display = np.vstack((result_image, info_bar))
        
        cv2.namedWindow('RGB Color Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('RGB Color Detection', combined_display)
        cv2.resizeWindow('RGB Color Detection', 800, 600)
        
        cv2.waitKey(3)
        
        if self.blue_found and not self.at_optimal_distance and not self.blue_approach_mode:
            # If we see blue but aren't in approach mode yet, cancel current Nav2 goal
            self.cancel_current_goal()
            self.blue_approach_mode = True
            self.get_logger().info("Blue object detected - switching to direct approach mode")
            
        elif not self.blue_found and self.blue_approach_mode:
            # If we lost the blue object while in approach mode, go back to Nav2
            self.blue_approach_mode = False
            self.get_logger().info("Lost blue object - returning to Nav2 exploration")
            
        
    
    def navigation_timer_callback(self):
        # If task is already completed, just maintain the stop position
        if self.task_completed:
            self.cmd_vel_pub.publish(self.stop_twist)
            return
            
        # If we're at the optimal distance from blue, stop and complete the task
        if self.blue_found and self.at_optimal_distance:
            self.cmd_vel_pub.publish(self.stop_twist)
            self.cancel_current_goal()
            self.task_completed = True
            self.get_logger().info("***** TASK COMPLETED: STOPPED AT BLUE BOX *****")
            return
        
        # If we're in blue approach mode, handle direct control
        if self.blue_approach_mode:
            self.direct_blue_approach()
            return
        
        # Otherwise, use Nav2 for exploration if not already navigating
        if not self.navigating and not self.blue_approach_mode and not self.task_completed:
            self.navigate_to_next_exploration_point()
    
    def direct_blue_approach(self):
        if not self.blue_found:
            # Lost sight of blue, stop and return to Nav2
            self.cmd_vel_pub.publish(self.stop_twist)
            self.blue_approach_mode = False
            return
        
        cmd = Twist()
        
        if abs(self.blue_area - self.blue_optimal_area) < self.area_tolerance:
            # We're at the optimal distance, just stop
            self.cmd_vel_pub.publish(self.stop_twist)
            self.get_logger().info("At optimal distance from blue - stopping")
            return
            
        # Check if we're too close to the blue object
        if self.blue_area > self.blue_optimal_area + self.area_tolerance:
            # Too close, back up straight (no turning)
            cmd.linear.x = -0.15
            cmd.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info(f"Too close to blue (area: {self.blue_area}) - backing up")
            return
            
        # If we're too far, we need to approach the blue object
        image_center_x = self.frame_width // 2
        center_error = self.blue_center_x - image_center_x
        
        # Only adjust angle if the centering error is significant
        if abs(center_error) > 50:
            angular_scale = 0.002
            cmd.angular.z = -center_error * angular_scale
            
            # Limit maximum angular velocity to a slower value
            max_angular = 0.3
            cmd.angular.z = max(min(cmd.angular.z, max_angular), -max_angular)
            
            # Don't move while turning significantly
            cmd.linear.x = 0.0
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info(f"Centering blue object (error: {center_error})")
        else:
            # Blue is centered, move forward at a moderate speed
            cmd.linear.x = 0.15
            cmd.angular.z = 0.0 
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info(f"Moving toward blue object (area: {self.blue_area})")
    
    def navigate_to_next_exploration_point(self):
        if self.blue_approach_mode or self.navigating:
            return
        
        # Get the next exploration point
        point = self.exploration_points[self.current_point_index]
        self.get_logger().info(f"Navigating to exploration point {self.current_point_index}: [{point[0]:.2f}, {point[1]:.2f}]")
        
        # Set up the navigation goal
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        # Set the position
        goal_pose.pose.position.x = float(point[0])
        goal_pose.pose.position.y = float(point[1])
        goal_pose.pose.position.z = 0.0
        
        # Set the orientation
        # Create quaternion for the desired yaw
        # For a rotation around z axis by angle theta:
        # x = 0, y = 0, z = sin(theta/2), w = cos(theta/2)
        theta = point[2]  # The yaw angle
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = math.sin(theta / 2.0)
        goal_pose.pose.orientation.w = math.cos(theta / 2.0)
        
        # Create and send the NavigateToPose goal
        navigate_goal = NavigateToPose.Goal()
        navigate_goal.pose = goal_pose
        
        self.navigating = True
        self.future_goal = self.nav_client.send_goal_async(navigate_goal)
        self.future_goal.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            self.navigating = False
            
            # Move to the next exploration point
            self.current_point_index = (self.current_point_index + 1) % len(self.exploration_points)
            return
        
        self.get_logger().info('Goal accepted')
        
        # Get the result future
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.result_callback)
    
    def result_callback(self, future):
        status = future.result().status
        
        if status == 4:  # SUCCEEDED
            self.get_logger().info('Navigation succeeded!')
        else:
            self.get_logger().info(f'Navigation failed with status: {status}')
        
        # Move to the next exploration point
        self.current_point_index = (self.current_point_index + 1) % len(self.exploration_points)
        
        # Navigation is complete
        self.navigating = False
    
    def cancel_current_goal(self):
        """Cancel any active navigation goal"""
        if self.navigating:
            # Try to cancel the current goal
            self.get_logger().info('Cancelling current navigation goal')
            
            # Send a stop command to ensure the robot stops immediately
            self.cmd_vel_pub.publish(self.stop_twist)
            
            self.navigating = False

def main():
    rclpy.init()
    
    node = Nav2ColorDetectionNode()
    
    def signal_handler(sig, frame):
        node.get_logger().info("Shutting down...")
        node.cmd_vel_pub.publish(node.stop_twist)
        cv2.destroyAllWindows()
        rclpy.shutdown()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_vel_pub.publish(node.stop_twist)
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
