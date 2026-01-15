#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
signal_processing.py - Importable signal processing functions and class structures.

Description
-----------
A collection of functions and class structures for discrete-time signal processing
tasks, including numerical differentiation and filtering. 

Current Features
----------------
1. Numerical Differentiation:
   - 'linear_diff': Computes the derivative of a signal using least-squares linear regression.
   - 'poly_diff': Computes the derivative using a local polynomial (Savitzky-Golay) fit.

2. Low-Pass Filtering:
   - First-order exponential moving average filters:
       * 'LowPassFilter'       : Fixed timestep low-pass filter.
       * 'LowPassFilter_VDT'   : Low-pass filter that handles variable timesteps.
   - Butterworth low-pass filters:
       * 'ButterworthLowPass'          : First-order fixed-timestep Butterworth filter.
       * 'ButterworthLowPass_VDT'      : First-order variable-timestep Butterworth filter.
       * 'ButterworthLowPass_VDT_2O'   : Second-order variable-timestep Butterworth filter.

Usage
-----
Import specific functions or classes as needed, e.g.:

    from signal_processing import linear_diff, LowPassFilter, ButterworthLowPass_VDT

Notes
-----
- Designed for discrete-time signals and real-time filtering applications.
- Variable timestep filters dynamically adjust smoothing based on elapsed time.
- All classes and functions are written for easy extension to additional filtering
  or signal processing operations in the future.

Author
------
Xander D. Mosley  
Email: XanderDMosley.Engineer@gmail.com  
Date: 30 Oct 2025
"""


import numpy as np
import warnings


__all__ = ['linear_diff', 'poly_diff',
           'LowPassFilter', 'LowPassFilter_VDT',
           'ButterworthLowPass', 'ButterworthLowPass_VDT', 'ButterworthLowPass_VDT_2O']
__author__ = "Xander D Mosley"
__email__ = "XanderDMosley.Engineer@gmail.com"


def linear_diff(
        time: np.ndarray,
        data: np.ndarray
        ) -> float:
    """
    Estimates the derivative of 'data' with respect to 'time' using
    a least-squares linear regression - originally designed for use
    over six samples.
    
    Parameters
    ----------
    time : np.ndarray
        1D array of time values (length >= 2).
    data : np.ndarray
        1D array of data values, same length as 'time'.
        
    Returns
    -------
    float
        Estimated derivative (slope) of data with respect to time.

    Author
    ------
    Xander D. Mosley

    History
    -------
    30 Oct 2025 - Created, XDM.
    """
    time = np.asarray(time, dtype=float).ravel()
    data = np.asarray(data, dtype=float).ravel()
    n_samples = time.size

    if time.size != data.size:
        raise ValueError("Arguments 'time' and 'data' must have the same length.")
    if n_samples < 2:
        raise ValueError("At least two points are required for differentiation")
        
    sum_xt = np.dot(data, time)
    sum_t = time.sum()
    sum_x = data.sum()
    sum_t2 = np.dot(time, time)
    
    denominator = (n_samples * sum_t2) - (sum_t ** 2)
    if (denominator == 0):
        raise ZeroDivisionError("Denominator in derivative computation is zero.")
        
    numerator = (n_samples * sum_xt) - (sum_x * sum_t)
        
    return numerator / denominator

def poly_diff(
        time: np.ndarray,
        data: np.ndarray,
        polyorder: int = 3,
        eval_point: str = "center"
        ) -> float:
    """
    Estimates the first derivative of 'data' with respect to 'time' using a
    local least-squares polynomial fit, following the Savitzky-Golay
    differentiation method.

    A polynomial of degree 'polyorder' is fit to the provided samples, and the
    derivative is evaluated at a specified point within the window.

    Parameters
    ----------
    time : np.ndarray
        1D array of time values (length > polyorder).
    data : np.ndarray
        1D array of data values, same length as 'time'.
    polyorder : int, optional
        Order of the polynomial to fit. Must be less than the number of samples.
        Default is 3.
    eval_point : {'start', 'center', 'end'}, optional
        Location within the window where the derivative is evaluated:
        - 'start': derivative at the first sample,
        - 'center': derivative at the midpoint sample,
        - 'end': derivative at the last sample.
        Default is 'center'.

    Returns
    -------
    float
        Estimated first derivative of 'data' with respect to 'time' at the
        specified evaluation point.

    Raises
    ------
    ValueError
        If 'time' and 'data' lengths differ.
    ValueError
        If 'polyorder' is not a valid integer.
    ValueError
        If 'eval_point' is not one of {'start', 'center', 'end'}.

    Notes
    -----
    - Using a 4th order polynomial starts to fit to noise.

    Author
    ------
    Xander D. Mosley

    History
    -------
    3 Nov 2025 - Created, XDM.
    """
    time = np.asarray(time, dtype=float).ravel()
    data = np.asarray(data, dtype=float).ravel()

    if time.size != data.size:
        raise ValueError("Arguments 'time' and 'data' must have the same length.")
    if not isinstance(polyorder, int) or isinstance(polyorder, bool):
        raise ValueError("Argument 'polyorder' must be an integer.")
    if time.size <= polyorder:
        raise ValueError("Number of samples must exceed polynomial order.")

    if eval_point == "center":
        eval_idx = len(time) // 2   # TODO: Ensure this is the center of five data points. Center of six data points?
    elif eval_point == "start":
        eval_idx = 0
    elif eval_point == "end":
        eval_idx = -1
    else:
        raise ValueError("eval_point must be 'start', 'center', or 'end'.")
    
    shifted_time = time - time[eval_idx]
    A = np.vander(shifted_time, N=polyorder + 1, increasing=True)

    coeffs, *_ = np.linalg.lstsq(A, data, rcond=None)
    deriv_coeffs = np.array([i * coeffs[i] for i in range(1, len(coeffs))])
    derivative = np.polyval(deriv_coeffs[::-1], 0.0)
    
    return np.float64(derivative)


class LowPassFilter:
    """
    First-order low-pass exponential moving average filter.

    This filter smooths noisy signals by applying a first-order low-pass
    filter, which attenuates high-frequency components while preserving
    low-frequency trends.

    Attributes
    ----------
    alpha : float
        Filter coefficient computed from cutoff frequency and timestep.
    filtered_value : float
        The current filtered value of the signal.

    Author
    ------
    Xander D. Mosley

    History
    -------
    4 Nov 2025 - Created, XDM.
    """
    def __init__(
            self,
            cutoff_frequency: float,
            dt: float,
            initial_value: float = 0.0
            ) -> None:
        """
        Initialize the low-pass filter.

        Parameters
        ----------
        cutoff_frequency : float
            The cutoff frequency of the filter in Hz. Determines how quickly
            the filter responds to changes in the input signal.
        dt : float
            The sampling interval in seconds.
        initial_value : float, optional
            Initial value for the filtered signal, by default 0.0.

        Notes
        -----
        The filter coefficient alpha is computed as:
            alpha = 1 - exp(-2 * pi * cutoff_frequency * dt)
        """
        self.alpha = 1 - np.exp(-2 * np.pi * cutoff_frequency * dt)
        self.filtered_value = initial_value

    def update(self, new_value: float) -> float:
        """
        Update the filter with a new input value and return the filtered output.

        Parameters
        ----------
        new_value : float
            The new raw input value to be filtered.

        Returns
        -------
        float
            The updated filtered value.

        Notes
        -----
        The filter applies the formula:
            filtered_value = alpha * new_value + (1 - alpha) * filtered_value
        which implements a first-order low-pass exponential moving average.
        """

        self.filtered_value = (self.alpha * new_value) + ((1 - self.alpha) * self.filtered_value)
        return self.filtered_value

class LowPassFilter_VDT:
    """
    First-order low-pass exponential moving average filter with variable time steps.

    This filter smooths noisy signals while accounting for variable sampling intervals.
    Unlike a standard low-pass filter with fixed timestep, this filter dynamically
    adapts the smoothing factor based on the observed time step (dt).

    Attributes
    ----------
    smoothing_factor : float
        Factor used to smooth changes in the observed timestep.
    average_dt : float
        Exponentially smoothed average of the variable time steps.
    fc : float
        Cutoff frequency of the filter in Hz.
    filtered_value : float
        Current filtered value of the signal.

    Author
    ------
    Xander D. Mosley

    History
    -------
    4 Nov 2025 - Created, XDM.
    """
    def __init__(
            self,
            cutoff_frequency: float,
            num_dts: int = 1,
            initial_value: float = 0.0
            ) -> None:
        """
        Initialize the variable-time-step low-pass filter.

        Parameters
        ----------
        cutoff_frequency : float
            The cutoff frequency of the filter in Hz.
        num_dts : int, optional
            Number of timesteps used for smoothing the average dt, by default 1.
        initial_value : float, optional
            Initial value for the filtered signal, by default 0.0.

        Notes
        -----
        The smoothing_factor for the average dt is computed as:
            smoothing_factor = 2 / (1 + num_dts)
        """
        num_dts = num_dts
        self.smoothing_factor = 2 / (1 + num_dts)
        self.average_dt = 0.0
        self.fc = cutoff_frequency
        self.filtered_value = initial_value

    def update(self, new_value: float, dt: float) -> float:
        """
        Update the filter with a new input value and time step, returning the filtered output.

        Parameters
        ----------
        new_value : float
            The new raw input value to be filtered.
        dt : float
            The elapsed time since the last update in seconds.

        Returns
        -------
        float
            The updated filtered value.

        Notes
        -----
        The filter dynamically computes alpha based on the smoothed average dt:
            average_dt = dt * smoothing_factor + (1 - smoothing_factor) * average_dt
            alpha = 1 - exp(-2 * pi * fc * average_dt)
            filtered_value = alpha * new_value + (1 - alpha) * filtered_value
        This allows the filter to handle variable update intervals.
        """
        self.average_dt = (dt * self.smoothing_factor) + ((1 - self.smoothing_factor) * self.average_dt)
        alpha = 1 - np.exp(-2 * np.pi * self.fc * self.average_dt)
        self.filtered_value = (alpha * new_value) + ((1 - alpha) * self.filtered_value)
        return self.filtered_value


class ButterworthLowPass:
    """
    First-order low-pass Butterworth filter.

    This filter provides a smooth signal by attenuating high-frequency components,
    based on a first-order Butterworth design. It is designed for discrete-time signals
    with a fixed sampling interval.

    Attributes
    ----------
    b0 : float
        Filter numerator coefficient for the current input.
    b1 : float
        Filter numerator coefficient for the previous input.
    a1 : float
        Filter denominator coefficient for the previous output.
    y_filtered : float
        Current filtered output value.
    x_previous : float
        Previous input value, used in filter recursion.

    Author
    ------
    Xander D. Mosley

    History
    -------
    6 Nov 2025 - Created, XDM.
    """
    def __init__(self, cutoff_frequency: float, dt: float):
        """
        Initialize the first-order Butterworth low-pass filter.

        Parameters
        ----------
        cutoff_frequency : float
            Desired cutoff frequency of the filter in Hz.
        dt : float
            Sampling interval of the input signal in seconds.

        Notes
        -----
        - The cutoff frequency is automatically clamped to be below the Nyquist
          frequency (0.5 / dt) to avoid instability.
        - Filter coefficients are calculated using a bilinear transform approach.
        - Warning is printed if the requested cutoff frequency exceeds 0.45 / dt.
        """
        fc = cutoff_frequency
        self.y_filtered = 0.0
        self.x_previous = 0.0

        fc_safe = min(fc, 0.45 / dt)
        if fc > (0.45 / dt):
            print("Warning: Cutoff frequency too high; clamped to 0.45 * fs.")
        gamma = np.tan(np.pi * fc_safe * dt)

        b0_prime = gamma
        b1_prime = b0_prime
        a1_prime = gamma - 1
        D = (gamma ** 2) + (np.sqrt(2) * gamma) + 1
        self.b0 = b0_prime / D
        self.b1 = b1_prime / D
        self.a1 = a1_prime / D

    def update(self, x_new: float):
        """
        Update the filter with a new input value and return the filtered output.

        Parameters
        ----------
        x_new : float
            The new raw input value to be filtered.

        Returns
        -------
        float
            The updated filtered output value.

        Notes
        -----
        The filter uses the difference equation:
            y[n] = b0 * x[n] + b1 * x[n-1] - a1 * y[n-1]
        where y[n] is the current output, x[n] is the current input, and x[n-1], y[n-1]
        are the previous input and output values, respectively.
        """
        y_new = (self.b0 * x_new) + (self.b1 * self.x_previous) - (self.a1 * self.y_filtered)
        self.x_previous = x_new
        self.y_filtered = y_new

        return y_new

class ButterworthLowPass_VDT:
    """
    First-order low-pass Butterworth filter with variable time steps.

    This filter smooths noisy signals while handling variable sampling intervals (dt).
    The cutoff frequency is adjusted dynamically to stay below the Nyquist limit for
    the given timestep, ensuring stability.

    Attributes
    ----------
    fc : float
        Desired cutoff frequency of the filter in Hz.
    y_filtered : float
        Current filtered output value.
    x_previous : float
        Previous input value, used in filter recursion.

    Author
    ------
    Xander D. Mosley

    History
    -------
    6 Nov 2025 - Created, XDM.
    """
    def __init__(self, cutoff_frequency: float):
        """
        Initialize the variable-time-step Butterworth low-pass filter.

        Parameters
        ----------
        cutoff_frequency : float
            Desired cutoff frequency of the filter in Hz.

        Notes
        -----
        - The filter does not require a fixed sampling interval, but adjusts
          coefficients dynamically based on the provided dt in each update.
        """
        self.fc = cutoff_frequency
        self.y_filtered = 0.0
        self.x_previous = 0.0

    def update(self, x_new: float, dt: float):
        """
        Update the filter with a new input value and variable timestep, returning
        the filtered output.

        Parameters
        ----------
        x_new : float
            The new raw input value to be filtered.
        dt : float
            The elapsed time since the last update in seconds.

        Returns
        -------
        float
            The updated filtered output value.

        Notes
        -----
        - The filter dynamically computes coefficients based on dt:
            fc_safe = min(fc, 0.45 / dt)
            gamma = tan(pi * fc_safe * dt)
        - The filter equation used is:
            y[n] = b0 * x[n] + b1 * x[n-1] - a1 * y[n-1]
        - A warning is printed if the cutoff frequency exceeds 0.45 / dt.
        """
        fc_safe = min(self.fc, 0.45 / dt)
        if self.fc > (0.45 / dt):
            print("Warning: Cutoff frequency too high; clamped to 0.45 * fs.")
        gamma = np.tan(np.pi * fc_safe * dt)

        b0_prime = gamma
        b1_prime = b0_prime
        a1_prime = gamma - 1
        D = (gamma ** 2) + (np.sqrt(2) * gamma) + 1
        b0 = b0_prime / D
        b1 = b1_prime / D
        a1 = a1_prime / D

        y_new = (b0 * x_new) + (b1 * self.x_previous) - (a1 * self.y_filtered)
        self.x_previous = x_new
        self.y_filtered = y_new

        return y_new

class ButterworthLowPass_VDT_2O:
    """
    Second-order low-pass Butterworth filter with variable time steps.

    This filter smooths noisy signals while handling variable sampling intervals (dt).
    It uses a second-order Butterworth design to achieve a steeper cutoff slope
    compared to first-order filters. Coefficients are dynamically adjusted
    based on the current timestep.

    Attributes
    ----------
    fc : float
        Desired cutoff frequency of the filter in Hz.
    y_filtered : list of float
        Previous filtered output values, used in the recursive filter equation.
        y_filtered[0] is the most recent output, y_filtered[1] is the one before that.
    x_previous : list of float
        Previous input values, used in the recursive filter equation.
        x_previous[0] is the most recent input, x_previous[1] is the one before that.

    Author
    ------
    Xander D. Mosley

    History
    -------
    6 Nov 2025 - Created, XDM.
    """
    def __init__(self, cutoff_frequency: float):
        """
        Initialize the second-order variable-time-step Butterworth low-pass filter.

        Parameters
        ----------
        cutoff_frequency : float
            Desired cutoff frequency of the filter in Hz.

        Notes
        -----
        - The filter does not require a fixed sampling interval, but adjusts
          coefficients dynamically based on dt provided in each update.
        - Initializes previous inputs and outputs to zero.
        """
        self.fc = cutoff_frequency
        self.y_filtered = [0.0, 0.0]
        self.x_previous = [0.0, 0.0]

    def update(self, x_new: float, dt: float):
        """
        Update the filter with a new input value and variable timestep, returning
        the filtered output.

        Parameters
        ----------
        x_new : float
            The new raw input value to be filtered.
        dt : float
            The elapsed time since the last update in seconds.

        Returns
        -------
        float
            The updated filtered output value.

        Notes
        -----
        - The cutoff frequency is dynamically clamped to be below the Nyquist limit:
            fc_safe = min(fc, 0.45 / dt)
        - The filter uses the difference equation:
            y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        - Previous input and output values are updated after each call.
        - A warning is printed if the requested cutoff frequency exceeds 0.45 / dt.
        """
        fc_safe = min(self.fc, 0.45 / dt)
        if self.fc > (0.45 / dt):
            print("Warning: Cutoff frequency too high; clamped to 0.45 * fs.")
        gamma = np.tan(np.pi * fc_safe * dt)

        b0_prime = gamma ** 2
        b1_prime = 2 * b0_prime
        b2_prime = b0_prime
        a1_prime = 2 * ((gamma ** 2) - 1)
        a2_prime = (gamma ** 2) - (np.sqrt(2) * gamma) + 1
        D = (gamma ** 2) + (np.sqrt(2) * gamma) + 1
        b0 = b0_prime / D
        b1 = b1_prime / D
        b2 = b2_prime / D
        a1 = a1_prime / D
        a2 = a2_prime / D

        y_new = (b0 * x_new) + (b1 * self.x_previous[0]) + (b2 * self.x_previous[1]) - (a1 * self.y_filtered[0]) - (a2 * self.y_filtered[1])
        self.x_previous[1] = self.x_previous[0]
        self.x_previous[0] = x_new
        self.y_filtered[1] = self.y_filtered[0]
        self.y_filtered[0] = y_new

        return y_new
    
    def current(self):
        return self.y_filtered[0]


# TODO: Add more filters types as testing continues.


if (__name__ == '__main__'):
    warnings.warn(
        "This script defines several functions and classes for"
        " signal processing, such as filtering and differentiating."
        "It is intented to be imported, not executed directly."
        "\n\tImport functions and class structures from this script using:\t"
        "from signal_processing import linear_diff, LowPassFilter, ButterworthLowPass_VDT"
        "\nMore functions and class structures are available within this script"
        " than the ones shown for example.",
        UserWarning)