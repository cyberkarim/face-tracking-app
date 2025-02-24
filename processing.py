import supervision as sv
import os
import numpy as np
import torch
import cv2
from typing import Callable
import argparse

Root_directory = os.getcwd()

model = torch.hub.load(r'yolov5', 'custom', path=Root_directory+"model/facenet.pt", source='local') 
model.conf = 0.70

#VIDEO_PATH = homewsluserMyworkPortfolioface-tracking-appvideos dataP2E_S5_C3.mp4
byte_tracker = sv.ByteTrack()
annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

def callback(frame: np.ndarray) -> np.ndarray:
    #img = frame.copy().resize((1, 3, 480, 640))
    results = model(frame,size=(1, 3, 640, 640))

    detections = sv.Detections.from_yolov5(results)
    detections = byte_tracker.update_with_detections(detections)
    if len(detections.confidence) == 0:
        return frame.copy()
    annotated_frame = frame.copy()
    annotated_frame = annotator.annotate(
    scene=annotated_frame,
    detections=detections
)

    labels = [
        f"#{detections.tracker_id[i]} {model.model.names[detections.class_id[i]]} {detections.confidence[i]:0.2f}"
       for i in range(0,len(detections.confidence))
     ]
    

    
    return label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

#sv.process_video(source_path=VIDEO_PATH, target_path=fP2E_S5_C3.mp4, callback=callback)
def process_stream(RTSP_ADDRESS,fps,callback :Callable):
    
    cap = cv2.VideoCapture(RTSP_ADDRESS,CAP_PROP_FPS=fps)
    index = 0
    while True:
        ret, frame = cap.read()
        if ret == True :
            index += 1
            result_frame = callback(frame, index)
            cv2.imshow(result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q') : # Press 'q' to exit
               break

    cv2.destroyAllWindows()

#target_pathfP2E_S5_C3.mp4
def process_video(
    VIDEO_PATH,
    target_path,
    callback :Callable) :
    sv.process_video(source_path=VIDEO_PATH, target_path=target_path, callback=callback)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script that greets you!")

     # Add the 'mode' argument with choices
    parser.add_argument(
        "mode",
        choices=["video", "stream"],
        help="Choose the mode 'video' (static proceccing) or 'stream' (real time processing)"
    )
    
    args, unknown = parser.parse_known_args()

    # Handle the chosen mode
    if args.mode == "video":
        print("Video mode selected")
         # Add video-specific arguments
        parser.add_argument(
            "--input", required=True, help="Path to the input video file"
        )
        parser.add_argument(
            "--output", help="Path to save the processed video (optional)"
        )
        args = parser.parse_args()
        process_video(args.input,args.output,callback=callback)
        
    elif args.mode == "stream":
        print("Stream mode selected")
        # Add stream-specific arguments
        parser.add_argument(
            "--RTSP", type=int, default=0, help="Camera index (default 0)"
        )
        parser.add_argument(
            "--fps", type=int, default=30, help="Frames per second for the stream"
        )
        # Parse the arguments
        args = parser.parse_args()
        process_stream(args.RTSP,args.fps,callback=callback)