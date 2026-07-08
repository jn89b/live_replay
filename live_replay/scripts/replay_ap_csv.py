#!/usr/bin/env python3
"""Replay ArduPilot CSV logs into the two topics consumed by InterpolateNode.

Publishes:
  - sensor_msgs/Imu on /ap/imu/experimental/data at imu_rate_hz
  - geographic_msgs/GeoPoseStamped on /ap/geopose/filtered at geopose_rate_hz

This is intended for the CSVs from BIN00000041:
  - BIN00000041_IMU(2).csv
  - 00000041_combined(1).csv
  - BIN00000041_XKQ.csv

Important:
  * The IMU CSV has 3 rows per TimeUS. This node keeps one row per timestamp
    by default, so the downstream node does not drop non-monotonic duplicate
    timestamps.
  * The combined CSV has multiple GPS-like entries close together. This node
    de-bursts them and keeps one GPS instance, then upsamples it.
  * XKQ is treated as q_nb: body-FRD -> local NED, with columns Q1,Q2,Q3,Q4
    in W,X,Y,Z order. It is converted into ROS ENU/FLU GeoPose orientation so
    InterpolateNode's geopose_enu_flu_to_ned_frd() conversion gives q_nb back.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import rclpy
from geographic_msgs.msg import GeoPoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu


NSEC_PER_SEC = 1_000_000_000


def read_csv_columns(path: str, columns: Iterable[str]) -> Dict[str, np.ndarray]:
    """Read only selected float columns from a CSV."""
    columns = list(columns)
    out: Dict[str, List[float]] = {c: [] for c in columns}

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{os.path.basename(path)} missing columns {missing}")

        for row in reader:
            for c in columns:
                out[c].append(float(row[c]))

    return {c: np.asarray(v, dtype=float) for c, v in out.items()}


def drop_duplicate_time_keep_first(
    data: Dict[str, np.ndarray],
    time_col: str,
) -> Dict[str, np.ndarray]:
    """Sort by time and keep the first row for each duplicate timestamp."""
    t = data[time_col]
    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    _, first_indices = np.unique(t_sorted, return_index=True)
    keep_sorted_indices = np.sort(first_indices)

    return {
        key: values[order][keep_sorted_indices]
        for key, values in data.items()
    }


def deburst_gps(
    gps: Dict[str, np.ndarray],
    instance_index: int,
    burst_gap_sec: float,
) -> Dict[str, np.ndarray]:
    """Keep one GPS sample from each near-simultaneous burst.

    The combined CSV does not contain a GPS instance column. In this log,
    GPS rows appear in small bursts separated by microseconds, then the next
    physical GPS update arrives roughly 0.2 s later. This function groups
    rows into those bursts and keeps one row from each group.
    """
    t = gps["GPS_t"]

    # First remove exact repeated GPS_t values caused by the combined table.
    _, first = np.unique(t, return_index=True)
    first = np.sort(first)
    gps = {k: v[first] for k, v in gps.items()}
    t = gps["GPS_t"]

    chosen: List[int] = []
    burst: List[int] = [0]

    for i in range(1, len(t)):
        if float(t[i] - t[i - 1]) <= burst_gap_sec:
            burst.append(i)
        else:
            chosen.append(burst[min(instance_index, len(burst) - 1)])
            burst = [i]

    if burst:
        chosen.append(burst[min(instance_index, len(burst) - 1)])

    chosen_arr = np.asarray(chosen, dtype=int)
    return {k: v[chosen_arr] for k, v in gps.items()}


def source_rate_report(t: np.ndarray) -> str:
    if len(t) < 2:
        return "n/a"
    dt = np.diff(t)
    dt = dt[dt > 0.0]
    if len(dt) == 0:
        return "n/a"
    med_dt = float(np.median(dt))
    return f"median dt={med_dt:.6f}s, median rate={1.0 / med_dt:.3f}Hz"


def segmented_target_times(
    src_t: np.ndarray,
    rate_hz: float,
    max_gap_sec: float,
) -> np.ndarray:
    """Build fixed-rate target times without interpolating across log gaps."""
    if len(src_t) < 2:
        return np.asarray([], dtype=float)

    period = 1.0 / float(rate_hz)
    breaks = np.where(np.diff(src_t) > max_gap_sec)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(src_t) - 1]

    grids: List[np.ndarray] = []
    for s, e in zip(starts, ends):
        if e <= s:
            continue
        t0 = float(src_t[s])
        t1 = float(src_t[e])
        if t1 <= t0:
            continue
        # Include t1 if it lands close to the grid.
        grid = np.arange(t0, t1 + 0.5 * period, period, dtype=float)
        grid = grid[grid <= t1 + 1e-9]
        if len(grid):
            grids.append(grid)

    if not grids:
        return np.asarray([], dtype=float)

    return np.concatenate(grids)


def interp_segmented(
    src_t: np.ndarray,
    src_y: np.ndarray,
    target_t: np.ndarray,
    max_gap_sec: float,
) -> np.ndarray:
    """Linear interpolation, returning NaN for targets inside source gaps."""
    out = np.full_like(target_t, np.nan, dtype=float)
    if len(src_t) < 2 or len(target_t) == 0:
        return out

    breaks = np.where(np.diff(src_t) > max_gap_sec)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(src_t) - 1]

    for s, e in zip(starts, ends):
        if e <= s:
            continue
        mask = (target_t >= src_t[s]) & (target_t <= src_t[e])
        if np.any(mask):
            out[mask] = np.interp(target_t[mask], src_t[s:e + 1], src_y[s:e + 1])

    return out


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def slerp_pair_wxyz(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Spherical interpolation between two WXYZ quaternions."""
    q0 = normalize_quat_wxyz(q0)
    q1 = normalize_quat_wxyz(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9995:
        return normalize_quat_wxyz(q0 + u * (q1 - q0))

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * u

    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return normalize_quat_wxyz(s0 * q0 + s1 * q1)


def slerp_series_wxyz(
    src_t: np.ndarray,
    q_wxyz: np.ndarray,
    target_t: np.ndarray,
    max_gap_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """SLERP q_wxyz onto target_t.

    Returns (target_quats, valid_mask).
    """
    out = np.zeros((len(target_t), 4), dtype=float)
    valid = np.zeros(len(target_t), dtype=bool)

    if len(src_t) < 2:
        return out, valid

    right_indices = np.searchsorted(src_t, target_t, side="right")

    for i, right in enumerate(right_indices):
        if right <= 0:
            continue
        if right >= len(src_t):
            # Allow exact final endpoint.
            if abs(float(target_t[i] - src_t[-1])) < 1e-9:
                out[i] = normalize_quat_wxyz(q_wxyz[-1])
                valid[i] = True
            continue

        left = right - 1
        dt = float(src_t[right] - src_t[left])
        if dt <= 0.0 or dt > max_gap_sec:
            continue

        u = float((target_t[i] - src_t[left]) / dt)
        u = max(0.0, min(1.0, u))
        out[i] = slerp_pair_wxyz(q_wxyz[left], q_wxyz[right], u)
        valid[i] = True

    return out, valid


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(q)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a WXYZ quaternion."""
    m00, m01, m02 = r[0]
    m10, m11, m12 = r[1]
    m20, m21, m22 = r[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return normalize_quat_wxyz(np.asarray([w, x, y, z], dtype=float))


def q_nb_wxyz_to_ros_enu_flu_xyzw(q_nb: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert q_nb WXYZ to ROS GeoPose ENU/FLU XYZW.

    q_nb maps body FRD -> navigation NED.
    ROS GeoPose orientation maps body FLU -> world ENU.
    """
    r_nb = quat_wxyz_to_matrix(q_nb)

    # v_enu = T_ned_to_enu * v_ned, where [E,N,U] = [E,N,-D].
    t_ned_to_enu = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )

    # v_frd = B_flu_to_frd * v_flu, where [F,R,D] = [F,-L,-U].
    b_flu_to_frd = np.diag([1.0, -1.0, -1.0])

    r_enu_flu = t_ned_to_enu @ r_nb @ b_flu_to_frd
    q_wxyz = matrix_to_quat_wxyz(r_enu_flu)
    w, x, y, z = q_wxyz
    return float(x), float(y), float(z), float(w)


def seconds_to_stamp_fields(stamp_s: float) -> Tuple[int, int]:
    stamp_ns = int(round(stamp_s * NSEC_PER_SEC))
    return int(stamp_ns // NSEC_PER_SEC), int(stamp_ns % NSEC_PER_SEC)


class ReplayArduPilotCsvTopics(Node):
    def __init__(self) -> None:
        super().__init__("replay_ap_csv_topics")

        self.declare_parameter("csv_dir", "/develop_ws/src/live_replay/live_replay/live_replay/csv_files")
        self.declare_parameter("imu_csv", "BIN00000041_IMU.csv")
        self.declare_parameter("combined_csv", "00000041_combined.csv")
        self.declare_parameter("xkq_csv", "BIN00000041_XKQ.csv")

        self.declare_parameter("imu_topic", "/ap/imu/experimental/data")
        self.declare_parameter("geopose_topic", "/ap/geopose/filtered")

        self.declare_parameter("imu_rate_hz", 175.0)
        self.declare_parameter("geopose_rate_hz", 80.0)
        self.declare_parameter("replay_speed", 1.0)

        self.declare_parameter("stamp_offset_sec", 1.0)
        self.declare_parameter("max_source_gap_sec", 1.0)

        # For this combined CSV, 0 usually selects the first GPS in each burst.
        # Change to 1 if you want the other GPS instance.
        self.declare_parameter("gps_instance_index", 0)
        self.declare_parameter("gps_burst_gap_sec", 0.05)

        self.declare_parameter("imu_frame_id", "base_link_frd")
        self.declare_parameter("geopose_frame_id", "map")
        self.declare_parameter("qos_depth", 300)

        csv_dir = str(self.get_parameter("csv_dir").value)
        imu_csv = str(self.get_parameter("imu_csv").value)
        combined_csv = str(self.get_parameter("combined_csv").value)
        xkq_csv = str(self.get_parameter("xkq_csv").value)

        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.geopose_topic = str(self.get_parameter("geopose_topic").value)
        self.imu_rate_hz = float(self.get_parameter("imu_rate_hz").value)
        self.geopose_rate_hz = float(self.get_parameter("geopose_rate_hz").value)
        self.replay_speed = float(self.get_parameter("replay_speed").value)
        self.stamp_offset_sec = float(self.get_parameter("stamp_offset_sec").value)
        self.max_source_gap_sec = float(self.get_parameter("max_source_gap_sec").value)
        self.gps_instance_index = int(self.get_parameter("gps_instance_index").value)
        self.gps_burst_gap_sec = float(self.get_parameter("gps_burst_gap_sec").value)
        self.imu_frame_id = str(self.get_parameter("imu_frame_id").value)
        self.geopose_frame_id = str(self.get_parameter("geopose_frame_id").value)
        qos_depth = int(self.get_parameter("qos_depth").value)

        if self.imu_rate_hz <= 0.0:
            raise ValueError("imu_rate_hz must be > 0")
        if self.geopose_rate_hz <= 0.0:
            raise ValueError("geopose_rate_hz must be > 0")
        if self.replay_speed <= 0.0:
            raise ValueError("replay_speed must be > 0")

        imu_path = os.path.join(csv_dir, imu_csv)
        combined_path = os.path.join(csv_dir, combined_csv)
        xkq_path = os.path.join(csv_dir, xkq_csv)

        self.imu_samples = self._prepare_imu_samples(imu_path)
        self.geo_samples = self._prepare_geopose_samples(combined_path, xkq_path)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=qos_depth,
        )

        self.imu_pub = self.create_publisher(Imu, self.imu_topic, qos)
        self.geo_pub = self.create_publisher(GeoPoseStamped, self.geopose_topic, qos)

        self.imu_index = 0
        self.geo_index = 0

        self.imu_timer = self.create_timer(
            1.0 / (self.imu_rate_hz * self.replay_speed),
            self.publish_imu,
        )
        self.geo_timer = self.create_timer(
            1.0 / (self.geopose_rate_hz * self.replay_speed),
            self.publish_geopose,
        )

        self.get_logger().info(
            f"Publishing IMU {len(self.imu_samples['t'])} samples at {self.imu_rate_hz:.1f} Hz "
            f"to {self.imu_topic}"
        )
        self.get_logger().info(
            f"Publishing GeoPose {len(self.geo_samples['t'])} samples at {self.geopose_rate_hz:.1f} Hz "
            f"to {self.geopose_topic}"
        )

    def _prepare_imu_samples(self, path: str) -> Dict[str, np.ndarray]:
        raw = read_csv_columns(
            path,
            ["TimeUS", "AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "t"],
        )
        raw = drop_duplicate_time_keep_first(raw, "TimeUS")

        src_t = raw["t"]
        target_t = segmented_target_times(
            src_t,
            self.imu_rate_hz,
            self.max_source_gap_sec,
        )

        if len(target_t) == 0:
            raise ValueError("No IMU target samples were generated")

        out = {"t": target_t}
        for col in ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]:
            out[col] = interp_segmented(src_t, raw[col], target_t, self.max_source_gap_sec)

        valid = np.ones(len(target_t), dtype=bool)
        for col in ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]:
            valid &= np.isfinite(out[col])

        out = {k: v[valid] for k, v in out.items()}

        self.get_logger().info(
            f"IMU source: {len(raw['t'])} unique timestamps after duplicate removal; "
            f"{source_rate_report(src_t)}"
        )
        return out

    def _prepare_geopose_samples(
        self,
        combined_path: str,
        xkq_path: str,
    ) -> Dict[str, np.ndarray]:
        gps_raw = read_csv_columns(
            combined_path,
            ["GPS_t", "GPS_Lat", "GPS_Lng", "GPS_Alt", "GPS_Spd"],
        )
        gps = deburst_gps(
            gps_raw,
            instance_index=self.gps_instance_index,
            burst_gap_sec=self.gps_burst_gap_sec,
        )
        gps_t = gps["GPS_t"]

        xkq = read_csv_columns(
            xkq_path,
            ["TimeUS", "Q1", "Q2", "Q3", "Q4", "t"],
        )
        xkq = drop_duplicate_time_keep_first(xkq, "TimeUS")
        q_src = np.column_stack([xkq["Q1"], xkq["Q2"], xkq["Q3"], xkq["Q4"]])

        target_t = segmented_target_times(
            gps_t,
            self.geopose_rate_hz,
            self.max_source_gap_sec,
        )

        if len(target_t) == 0:
            raise ValueError("No GeoPose target samples were generated")

        lat = interp_segmented(gps_t, gps["GPS_Lat"], target_t, self.max_source_gap_sec)
        lng = interp_segmented(gps_t, gps["GPS_Lng"], target_t, self.max_source_gap_sec)
        alt = interp_segmented(gps_t, gps["GPS_Alt"], target_t, self.max_source_gap_sec)

        q_nb, q_valid = slerp_series_wxyz(
            xkq["t"],
            q_src,
            target_t,
            max_gap_sec=self.max_source_gap_sec,
        )

        valid = (
            np.isfinite(lat)
            & np.isfinite(lng)
            & np.isfinite(alt)
            & q_valid
        )

        target_t = target_t[valid]
        lat = lat[valid]
        lng = lng[valid]
        alt = alt[valid]
        q_nb = q_nb[valid]

        # Convert every q_nb WXYZ to GeoPose-compatible ROS ENU/FLU XYZW.
        q_xyzw = np.asarray(
            [q_nb_wxyz_to_ros_enu_flu_xyzw(q) for q in q_nb],
            dtype=float,
        )

        self.get_logger().info(
            f"GPS source: {len(gps_t)} samples after de-bursting; "
            f"{source_rate_report(gps_t)}; gps_instance_index={self.gps_instance_index}"
        )
        self.get_logger().info(
            f"XKQ source: {len(xkq['t'])} unique timestamps after duplicate removal; "
            f"{source_rate_report(xkq['t'])}"
        )

        return {
            "t": target_t,
            "lat": lat,
            "lng": lng,
            "alt": alt,
            "qx": q_xyzw[:, 0],
            "qy": q_xyzw[:, 1],
            "qz": q_xyzw[:, 2],
            "qw": q_xyzw[:, 3],
        }

    def _set_header_stamp(self, header, replay_t: float) -> None:
        # Add 1 second by default so the first message stamp is not zero.
        sec, nsec = seconds_to_stamp_fields(self.stamp_offset_sec + float(replay_t))
        header.stamp.sec = sec
        header.stamp.nanosec = nsec

    def publish_imu(self) -> None:
        if self.imu_index >= len(self.imu_samples["t"]):
            self.imu_timer.cancel()
            self.get_logger().info("Finished IMU replay")
            self._shutdown_if_done()
            return

        i = self.imu_index
        msg = Imu()
        self._set_header_stamp(msg.header, self.imu_samples["t"][i])
        msg.header.frame_id = self.imu_frame_id

        # InterpolateNode ignores the raw IMU orientation, but set it valid.
        msg.orientation.w = 1.0
        msg.orientation_covariance[0] = -1.0

        msg.linear_acceleration.x = float(self.imu_samples["AccX"][i])
        msg.linear_acceleration.y = float(self.imu_samples["AccY"][i])
        msg.linear_acceleration.z = float(self.imu_samples["AccZ"][i])

        msg.angular_velocity.x = float(self.imu_samples["GyrX"][i])
        msg.angular_velocity.y = float(self.imu_samples["GyrY"][i])
        msg.angular_velocity.z = float(self.imu_samples["GyrZ"][i])

        self.imu_pub.publish(msg)
        self.imu_index += 1

    def publish_geopose(self) -> None:
        if self.geo_index >= len(self.geo_samples["t"]):
            self.geo_timer.cancel()
            self.get_logger().info("Finished GeoPose replay")
            self._shutdown_if_done()
            return

        i = self.geo_index
        msg = GeoPoseStamped()
        self._set_header_stamp(msg.header, self.geo_samples["t"][i])
        msg.header.frame_id = self.geopose_frame_id

        msg.pose.position.latitude = float(self.geo_samples["lat"][i])
        msg.pose.position.longitude = float(self.geo_samples["lng"][i])
        msg.pose.position.altitude = float(self.geo_samples["alt"][i])

        msg.pose.orientation.x = float(self.geo_samples["qx"][i])
        msg.pose.orientation.y = float(self.geo_samples["qy"][i])
        msg.pose.orientation.z = float(self.geo_samples["qz"][i])
        msg.pose.orientation.w = float(self.geo_samples["qw"][i])

        self.geo_pub.publish(msg)
        self.geo_index += 1

    def _shutdown_if_done(self) -> None:
        if self.imu_index >= len(self.imu_samples["t"]) and self.geo_index >= len(self.geo_samples["t"]):
            self.get_logger().info("All replay streams finished. Shutting down.")
            rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ReplayArduPilotCsvTopics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
