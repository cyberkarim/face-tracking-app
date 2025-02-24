# face-tracking-app

Clone the repo

- create a conda virtual environement in order to store all project dependencies

- activate the environement

- pip install requirements.txt

I use YOLO version 5 in this project. Notebooks related to data preparation and model training are saved in the "processing training" folder.

## video processing

use relative paths that start from your current working directory to indicate the source video file to process and where to output it afterwards.
python processing.py video --input input file path --output output file path


## stream processing
python script.py stream --RTSP camera rtsp adress (0 for localhost webcam) --fps frame rate for processing
