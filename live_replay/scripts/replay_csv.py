#!/usr/bin/env python3

import math
import os
import threading
from re import S
import re

import numpy as np
import csv
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.publisher import Publisher

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Float64MultiArray, String
from drone_interfaces.msg import CtlTraj, Telem


class ReplayCSV(Node):
    def __init__(self, ns=''):
        super().__init__('replay_node')

        self.setup_vars()
        self.setup_pubs()

    def setup_vars(self):
        # Directory containing CSVs
        self.csv_dir = '/develop_ws/src/live_replay/live_replay/live_replay/csv_files'

        # Explicit list of CSVs you want replayed
        # NOTE: File names cannot start with numbers because the file names become variable names later in the script.
        self.requested_csvs = [
            'BIN00000091_IMU.csv',
            'BIN00000091_RCOU.csv',
        ]

        self.csv_streams = {}

        # ---- Validate requested files ----
        available_files = set(os.listdir(self.csv_dir))
        missing = [f for f in self.requested_csvs if f not in available_files]

        if missing:
            raise FileNotFoundError(
                f'Requested CSV files not found: {missing}'
            )

        # ---- Load and process each requested CSV ----
        for fname in self.requested_csvs:
            path = os.path.join(self.csv_dir, fname)

            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames


            if not rows or fieldnames is None or 't' not in fieldnames:
                raise ValueError(
                    f'{fname} must contain a "t" column and at least one row'
                )

            times = [float(r['t']) for r in rows]

            if len(times) < 2:
                raise ValueError(f'{fname} must contain at least two samples')

            start_time = times[0]
            end_time = times[-1]
            num_samples = len(rows)

            period = (end_time - start_time) / num_samples

            if period <= 0.0:
                raise ValueError(f'Invalid timing data in {fname}')

            data_columns = [c for c in fieldnames if c != 't']

            if not data_columns:
                raise ValueError(f'{fname} has no data columns')

            self.csv_streams[fname] = {
                'name': fname,
                'rows': rows,
                'columns': data_columns,
                'index': 0,
                'period': period,
                'publisher': None,
                'timer': None,
                'finished': False,
            }

        self.total_csvs = len(self.csv_streams)
        self.finished_csvs = 0

    def _strip_bin_prefix(self, fname: str) -> str:
        """
        Removes leading BIN########_ from a filename if present.
        """
        return re.sub(r'^BIN\d+_', '', fname)

    def setup_pubs(self):
        for fname, stream in self.csv_streams.items():
            clean_name = self._strip_bin_prefix(fname)
            base_name = clean_name.replace('.csv', '')

            topic_name = f'replay/{base_name}_data'

            stream['publisher'] = self.create_publisher(
                Float64MultiArray,
                topic_name,
                10
            )

            stream['timer'] = self.create_timer(
                stream['period'],
                lambda s=stream: self.publish_csv_step(s)
            )

    def publish_csv_step(self, stream):
        idx = stream['index']
        rows = stream['rows']

        if stream['finished']:
            return

        # End of CSV
        if idx >= len(rows):
            stream['finished'] = True
            stream['timer'].cancel()
            self.get_logger().info(
                f'Finished replaying CSV: {stream["name"]}'
            )

            # ---- Check if all CSVs are finished ----
            if all(s['finished'] for s in self.csv_streams.values()):
                self.get_logger().info(
                    'All CSVs finished. Shutting down replay node.'
                )

                # Destroy node and shutdown ROS
                self.destroy_node()
                rclpy.shutdown()
            return

        row = rows[idx]

        msg = Float64MultiArray()
        msg.data = [float(row[col]) for col in stream['columns']]

        stream['publisher'].publish(msg)
        stream['index'] += 1


def main(args=None):
    rclpy.init(args=args)
    replay_node = ReplayCSV()

    while rclpy.ok():
        try:
            rclpy.spin_once(replay_node, timeout_sec=0.1)

        except KeyboardInterrupt:
            break

    replay_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()