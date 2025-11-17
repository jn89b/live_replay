import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from pymavlink import DFReader
from typing import List, Optional, Dict


class FlightParser:
    def __init__(self,
                 log_path: str) -> None:
        self.log_path: str = log_path
        self.binary_log: str = DFReader.DFReader_binary(
            filename=log_path)
        # Peek what’s available:
        print("Message types present:", sorted(
            [fmt.name for fmt in self.binary_log.formats.values()]))
        print("Counts by type (non-zero):",
              {self.binary_log.id_to_name[i]: c for i, c in enumerate(self.binary_log.counts) if c > 0 and i in self.binary_log.id_to_name})
        # we do this to reset the log for future reads
        self.binary_log.rewind()

    def add_rel_time(self, df: pd.DataFrame, col="TimeUS") -> pd.DataFrame:
        """
        Returns an additional column to the dataframe that contains the relative time in units
        of seconds 

        Args:
            df (pd.DataFrame): The input dataframe to which the relative time column will be added.
            col (str): The name of the column containing the absolute time values.

        Returns:
            pd.DataFrame: The input dataframe with an additional column "t" containing the relative time in seconds.

        """
        if df is None or df.empty or col not in df:
            return df
        df = df.sort_values(col).copy()
        t0 = df[col].iloc[0]
        df["t"] = (df[col] - t0) / 1e6
        return df

    def get_desired_data(self,
                         types: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Args:
            types (List[str]): A list of message types to extract from the log.
            The names from the string come from the ardupilot documentation, 
            refer to https://ardupilot.org/plane/docs/common-downloading-and-analyzing-data-logs-in-mission-planner.html
            in the Message Details section of the link
        Returns:
            Dict[str, pd.DataFrame]: A dictionary of DataFrames, one for each message type.
        """
        if not types:
            raise ValueError("Types list cannot be empty.")

        rows: Dict[str, List[Dict[str, float]]] = {t: [] for t in types}
        while True:
            m = self.binary_log.recv_msg()
            if m is None:
                break
            t = m.get_type()
            if t in types:
                d = m.to_dict()
                d["TimeUS"] = getattr(m, "TimeUS", None)
                d["_t"] = getattr(m, "_timestamp", None)
                rows[t].append(d)

        dfs = {t: pd.DataFrame(rows[t]) for t in types}
        for k, df in dfs.items():
            dfs[k] = self.add_rel_time(df)

        return dfs
    
def pickle_desired_data(desired_data: Dict[str, pd.DataFrame], file_path: str) -> None:
    """
    Pickles the desired data dictionary to a specified file path.
    Data is saved as hashtables of dataframes for each message type.
    Args:
        desired_data (Dict[str, pd.DataFrame]): The dictionary of DataFrames to pickle.
        file_path (str): The file path where the pickled data will be saved.
    Returns:
        None
    """
    with open(file_path, 'wb') as f:
        pickle.dump(desired_data, f)


def generate_command_data(attitude_data: pd.DataFrame, 
                          throttle_data: pd.DataFrame,
                          dt:float = 0.05,
                          file_name: str ="command_data",
                          save_to_csv:bool=False) -> pd.DataFrame:
    """
    Generates a roll,pitch,yaw,throttle command dataframe based on the attitude and throttle data.
    
    Args:
        attitude_data (pd.DataFrame): DataFrame containing 't', 'DesRoll', '
            'DesPitch', 'DesYaw' columns.
        throttle_data (pd.DataFrame): DataFrame containing 't', 'ThO' columns
        dt (float): Time step for the command data.
        file_name (str): Name of the file to save the command data CSV.
        save_to_csv (bool): Whether to save the generated command data to a CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the generated command data with columns
            't', 'DesRoll', 'DesPitch', 'DesYaw', 'DesThrottle'.

    """
    len_data = len(attitude_data)
    time_command = np.arange(0, len_data * dt, dt)
    desired_roll = np.interp(attitude_data['t'], attitude_data['t'], attitude_data['DesRoll'])
    desired_pitch = np.interp(attitude_data['t'], attitude_data['t'], attitude_data['DesPitch'])
    desired_yaw = np.interp(attitude_data['t'], attitude_data['t'], attitude_data['DesYaw'])
    desired_throttle = np.interp(throttle_data['t'], throttle_data['t'], throttle_data['ThO'])
    # map the throttle based on time 
    mapped_throttle = np.interp(time_command, throttle_data['t'], desired_throttle)
    scaled_throttle = mapped_throttle / 100.0 # reduce to 0 to 1 range
    # regularize to the attitude time stamps
    command_data = pd.DataFrame({
        't': time_command,
        'DesRoll': desired_roll,
        'DesPitch': desired_pitch,
        'DesYaw': desired_yaw,
        'DesThrottle': scaled_throttle
    })
    # 
    if save_to_csv:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(current_dir, "maneuvers")
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, str(file_name))
        command_data.to_csv(filepath + ".csv", index=False)

    return command_data

def make_csv(file_name:str) -> None:
    """
    Turn this into a parser to generate the file we want for live replay
    """
    flight_parser: FlightParser = FlightParser(log_path=file_name + ".BIN")
    desired_data: Dict[str, pd.DataFrame] = flight_parser.get_desired_data(
        types=["GPS", "XKF1", "CTUN", "MODE", "IMU", "ATT", "AHR2", "RCIN", "RCOU", "CMD", "ARSP"])
    attitude_data = desired_data.get("ATT", pd.DataFrame())
    throttle_data = desired_data.get("CTUN", pd.DataFrame())
    command: pd.DataFrame = generate_command_data(
        attitude_data=attitude_data,
        throttle_data=throttle_data,
        dt=0.05,
        file_name=file_name+"_command_data",
        save_to_csv=True
    )
    
if __name__ == "__main__":    
    make_csv("binaries/00000050 - GNC_SID_(EmergencyLanding)")
        
    