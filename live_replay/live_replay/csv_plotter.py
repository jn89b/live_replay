#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from signal_processing import ButterworthLowPass_VDT_2O, poly_diff


def plot_csv(csv_file, time_col, data_cols,
             figure_title="CSV Data Plot",
             figure_size=(10, 3)):
    """
    Plot CSV data with one subplot per data column.

    Parameters
    ----------
    csv_file : str
        Path to CSV file
    time_col : str
        Name of the time column
    data_cols : list[str]
        List of data column names (one subplot per column)
    figure_title : str
        Title for the entire figure
    figure_size : tuple(float, float)
        (width, height_per_subplot)
    """

    # Load data
    df = pd.read_csv(csv_file)

    # Validate columns
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in CSV.")

    for col in data_cols:
        if col not in df.columns:
            raise ValueError(f"Data column '{col}' not found in CSV.")

    # Create subplots
    n = len(data_cols)
    fig, axes = plt.subplots(
        n,
        1,
        sharex=True,
        figsize=(figure_size[0], figure_size[1] * n)
    )

    # Ensure axes is iterable for single subplot
    if n == 1:
        axes = [axes]

    # Plot each data column
    for ax, col in zip(axes, data_cols):
        ax.scatter(df[time_col], df[col], alpha=0.7, marker='.')
        ax.plot(df[time_col], df[col], alpha=0.3)
        ax.set_ylabel(col)
        ax.grid(True)

    axes[-1].set_xlabel(time_col)
    fig.suptitle(figure_title, fontsize=14)

    plt.tight_layout()
    plt.show()


def apply_butterworth_vdt(
    time: np.ndarray,
    data: np.ndarray,
    cutoff_frequency: float,
    ):
    """
    Apply a stateful second-order variable-dt Butterworth low-pass filter
    to NumPy arrays.

    Parameters
    ----------
    time : np.ndarray
        1D array of timestamps (seconds), strictly increasing.
    data : np.ndarray
        1D array of signal values.
    cutoff_frequency : float
        Low-pass cutoff frequency in Hz.

    Returns
    -------
    np.ndarray
        Filtered signal array.
    """
    time = np.asarray(time, dtype=float).ravel()
    data = np.asarray(data, dtype=float).ravel()

    if time.size != data.size:
        raise ValueError("time and data must have the same length.")

    filt = ButterworthLowPass_VDT_2O(cutoff_frequency)

    filtered = np.empty_like(data, dtype=float)

    prev_t = None

    for i, (t, x) in enumerate(zip(time, data)):
        if prev_t is None:
            filtered[i] = x
        else:
            dt = t - prev_t
            if dt > 0:
                filtered[i] = filt.update(x, dt)
            else:
                filtered[i] = filtered[i - 1]
        prev_t = t

    return filtered

def differentiate_signal_poly(
    time: np.ndarray,
    data: np.ndarray,
    polyorder: int = 3,
    window_size: int | None = None,
    eval_point: str = "center",
    ):
    """
    Differentiate an entire signal using local polynomial least-squares
    differentiation (Savitzky-Golay style).

    Parameters
    ----------
    time : np.ndarray
        1D array of time values.
    data : np.ndarray
        1D array of signal values (e.g., filtered data).
    polyorder : int, optional
        Polynomial order for local fit. Default is 3.
    window_size : int, optional
        Number of points in each local window. If None, uses polyorder + 2.
        Must be > polyorder.
    eval_point : {'start', 'center', 'end'}, optional
        Where to evaluate the derivative within the window.

    Returns
    -------
    np.ndarray
        Estimated derivative of `data` with respect to `time`.
    """
    time = np.asarray(time, dtype=float)
    data = np.asarray(data, dtype=float)

    if time.size != data.size:
        raise ValueError("time and data must have the same length.")

    if window_size is None:
        window_size = polyorder + 2

    if window_size <= polyorder:
        raise ValueError("window_size must be greater than polyorder.")

    n = len(time)
    derivative = np.full(n, np.nan)

    # Determine evaluation index offset
    if eval_point == "center":
        offset = window_size // 2
    elif eval_point == "start":
        offset = 0
    elif eval_point == "end":
        offset = window_size - 1
    else:
        raise ValueError("eval_point must be 'start', 'center', or 'end'.")

    for i in range(n):
        start = i - offset
        end = start + window_size

        if start < 0 or end > n:
            continue  # leave as NaN at boundaries

        t_window = time[start:end]
        d_window = data[start:end]

        derivative[i] = poly_diff(
            t_window,
            d_window,
            polyorder=polyorder,
            eval_point=eval_point,
        )

    return derivative

def add_accels(csv_directory, csv_name):
    df = pd.read_csv(csv_directory + csv_name)
    columns_to_keep = [
        "TimeUS", "IMU_t",
        "IMU_AccX", "IMU_AccY", "IMU_AccZ", "IMU_GyrX", "IMU_GyrY", "IMU_GyrZ",
        "ATT_Roll", "ATT_Pitch", "ATT_Yaw", "ATT_DesRoll", "ATT_DesPitch", "ATT_DesYaw", 
        "RCOU_C1", "RCOU_C2", "RCOU_C3", "RCOU_C4",
        "GPS_Lat", "GPS_Lng", "GPS_Alt", "GPS_Spd",
        "CTUN_ThO"
    ]
    new_df = df.loc[::2, columns_to_keep]

    time = "IMU_t"
    GyrX_Filt = apply_butterworth_vdt(df[time].to_numpy(), df["IMU_GyrX"].to_numpy(), cutoff_frequency=1.54)
    GyrY_Filt = apply_butterworth_vdt(df[time].to_numpy(), df["IMU_GyrY"].to_numpy(), cutoff_frequency=1.54)
    GyrZ_Filt = apply_butterworth_vdt(df[time].to_numpy(), df["IMU_GyrZ"].to_numpy(), cutoff_frequency=1.54)
    Diff_GyrX = differentiate_signal_poly(df[time].to_numpy(), np.asarray(GyrX_Filt))
    Diff_GyrY = differentiate_signal_poly(df[time].to_numpy(), np.asarray(GyrY_Filt))
    Diff_GyrZ = differentiate_signal_poly(df[time].to_numpy(), np.asarray(GyrZ_Filt))
    Diff_GyrX_Filt = apply_butterworth_vdt(df[time].to_numpy(), np.asarray(Diff_GyrX), cutoff_frequency=1.54)
    Diff_GyrY_Filt = apply_butterworth_vdt(df[time].to_numpy(), np.asarray(Diff_GyrY), cutoff_frequency=1.54)
    Diff_GyrZ_Filt = apply_butterworth_vdt(df[time].to_numpy(), np.asarray(Diff_GyrZ), cutoff_frequency=1.54)

    new_df["DIFF_GyrX"] = Diff_GyrX_Filt[::2]
    new_df["DIFF_GyrY"] = Diff_GyrY_Filt[::2]
    new_df["DIFF_GyrZ"] = Diff_GyrZ_Filt[::2]

    new_df.to_csv(csv_directory + "added_accels.csv", index=False)


if __name__ == "__main__":
    csv_folder = "/develop_ws/src/live_replay/live_replay/live_replay/csv_files/"
    csv_file = "00000091_combined.csv"

    # add_accels(
    #     csv_directory=csv_folder,
    #     csv_name=csv_file
    # )
    
    csv_file = "added_accels.csv"
    time_col = "IMU_t"
    data_cols = [
        # "IMU_AccX",
        # "IMU_AccY",
        # "IMU_AccZ",
        "IMU_GyrX",
        "IMU_GyrY",
        "IMU_GyrZ",

        # "ATT_Roll",
        # "ATT_Pitch",
        # "ATT_Yaw",
        # "ATT_DesRoll",
        # "ATT_DesPitch",
        # "ATT_DesYaw",

        # "RCOU_C1",
        # "RCOU_C2",
        # "RCOU_C3",
        # "RCOU_C4",

        # "GPS_Lat",
        # "GPS_Lng",
        "GPS_Alt",
        # "GPS_Spd",

        # "CTUN_ThO",

        # "DIFF_GyrX",
        # "DIFF_GyrY",
        # "DIFF_GyrZ",
    ]

    plot_csv(
        csv_file=csv_folder + csv_file,
        time_col=time_col,
        data_cols=data_cols,
    )