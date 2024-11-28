from ultralytics import YOLO
import cv2
import numpy as np
import util
from add_missing_data import*
from sort.tracker import SortTracker
tracker = SortTracker(1)
from util import get_car, read_license_plate, write_csv
from visualize import*
import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

results = {}

# load models
coco_model = YOLO('yolov8s.pt')
license_plate_detector = YOLO('license_plate.pt')

## name the video of traffic "input.mp4"in work directory
input_path ="input.mp4"
# load video
cap = cv2.VideoCapture(input_path)

vehicles = [2, 3, 5, 7]

# Read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        
        # Detect vehicles using YOLO
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score,class_id])

        # Convert detections to numpy array before passing to tracker
        detections_ = np.array(detections_)
        # print (detections_)

        # Track vehicles
        track_ids = tracker.update(detections_,None)

        # Detect license plates using YOLO
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            
            xcar1, ycar1, xcar2, ycar2,car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # Crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# Write results to CSV
write_csv(results, './test.csv')

# Load the CSV file
with open('test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)

result_ = pd.read_csv('./test_interpolated.csv')


give_result(result_,input_path)
