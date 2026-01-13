# Live Replay
- This repo is used to replay the commands from the Binary files from Ardupilot
- To do place your binary files inside the live_replay/binaries repo:
- Use the generate_csv file to generate the csv to    


## Generating a Dataset
- To create a dataset you need to have .BIN files from Ardupilot, save it in the "binaries/" directory it doesn't exist make one
- Now in the live_replay/generate_csv.py change the directory of the make_csv(file_path) to your file_path and run it
- This will generate *.csv files based on what you have. 