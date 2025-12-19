#!/usr/bin/env python3

import os
import rclpy
import math
import numpy as np
import mavros
import pandas as pd
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from nav_msgs.msg import Odometry
from drone_interfaces.msg import Telem, CtlTraj
from re import S
from typing import List
from mavros.base import SENSOR_QOS

"""
For this application we will be sending roll, pitch yaw commands to the drone
"""

class ReplayCommandNode(Node):
    """
    GOAL want to publish a roll,pitch,yaw trajectory to the drone to get
    to the target location
    HINTS Not in order
    - Remember the current coordinate system is in ENU need to convert to NED
    - Might need to add some safety checks
    - Need to calculate something to get the roll, pitch, yaw commands
        - Yaw and roll control the lateral motion
        - Pitch control the vertical motion  
    - Need to subscribe to something else besides the mavros state
    """

    def __init__(self, 
                 csv_des: str,
                 ns=''):
        super().__init__('pub_example')

        self.trajectory_publisher: Publisher = self.create_publisher(
            CtlTraj, 'trajectory', 10)
        
        self.csv_command_file = csv_des
        
        if not os.path.isfile(self.csv_command_file):
            raise FileNotFoundError(f"CSV file {self.csv_command_file} not found.")
        self.data_frame: pd.DataFrame = pd.read_csv(self.csv_command_file)
        #self.data_frame: pd.DataFrame = self.get_nonzero_throttle_commands(0.35)
        # self.data_frame: pd.DataFrame = self.get_time_desired_commands(2250.0)
        self.parse_out_commands()

        self.publish_freq: int = 25.0  # 25 Hz
        self.timer_period: float = 1.0 / self.publish_freq  # seconds
        self.repeat: bool = False  # whether to repeat the commands after finishing
        
        self.counter:int = 0
        self.command_timer = self.create_timer(
            self.timer_period, self.command_callback)
    
    def get_time_desired_commands(self, desired_time:float) -> pd.DataFrame:
        """
        Returns a DataFrame of commands where the time is greater than or equal to the desired time.
        """
        idx = self.data_frame[self.data_frame["t"] >= desired_time].index

        if len(idx) == 0:
            print("No rows where time is greater than or equal to desired time")
        else:
            first = idx[0]
            df_filtered = self.data_frame.loc[first:]
            print(df_filtered)
            
        return df_filtered
    
    def get_nonzero_throttle_commands(self, desired_value:float) -> pd.DataFrame:
        """
        Returns a DataFrame of commands where the throttle is non-zero.
        """
        idx = self.data_frame[self.data_frame["DesThrottle"] == desired_value].index

        if len(idx) == 0:
            print("No rows where DesThrottle is non-zero")
        else:
            first = idx[0]
            df_filtered = self.data_frame.loc[first:]
            print(df_filtered)
            
        return df_filtered
        
    def parse_out_commands(self) -> None:
        """
        Checks out the data frame and parses out the roll, pitch, yaw, throttle commands
        into lists for publishing
        """
        self.roll_commands: List[float] = self.data_frame['DesRoll'].tolist()
        self.pitch_commands: List[float] = self.data_frame['DesPitch'].tolist()
        self.yaw_commands: List[float] = self.data_frame['DesYaw'].tolist()
        self.throttle_commands: List[float] = self.data_frame['DesThrottle'].tolist()
        
        self.roll_commands = [math.radians(angle) for angle in self.roll_commands]
        self.pitch_commands = [math.radians(angle) for angle in self.pitch_commands]
        self.yaw_commands = [math.radians(angle) for angle in self.yaw_commands]
        
    def command_callback(self) -> None:
        """
        Publishes the roll, pitch, yaw, throttle commands at the specified frequency
        and iterates through the command list
        """
        trajectory = CtlTraj()

        if self.counter >= len(self.roll_commands):
            if self.repeat:
                self.counter = 0
            else:
                self.get_logger().info("All commands published.")
            return

        trajectory.roll = [float(self.roll_commands[self.counter])]
        # negative pitch since we need to invert for NED
        trim_pitch: float = np.deg2rad(1.5)
        trajectory.pitch = [float(self.pitch_commands[self.counter]) + trim_pitch       ]
        trajectory.yaw = [float(self.yaw_commands[self.counter])]
        trajectory.thrust = [float(self.throttle_commands[self.counter])]
        self.publish_trajectory(trajectory)
        self.get_logger().info(f"Published command {self.counter}: "
                               f"Roll: {trajectory.roll}, "
                               f"Pitch: {trajectory.pitch}, "
                               f"Yaw: {trajectory.yaw}, "
                               f"Throttle: {trajectory.thrust}")
        self.counter += 1

    def publish_trajectory(self, trajectory: CtlTraj) -> None:
        """
        Publishes the trajectory
        """
        self.trajectory_publisher.publish(trajectory)


def main() -> None:
    rclpy.init()
    file_dir = "/develop_ws/src/live_replay/live_replay/live_replay/maneuvers/00000050-GNC_SID_(EmergencyLanding).BIN_command_data.csv"
    command_publisher: ReplayCommandNode = ReplayCommandNode(csv_des=file_dir)
    while rclpy.ok():
        try:
            rclpy.spin(command_publisher)
        except KeyboardInterrupt:
            command_publisher.get_logger().info('Keyboard Interrupt')
            break

    command_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
