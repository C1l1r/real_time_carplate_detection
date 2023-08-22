from ultralytics import YOLO
import cv2
import datetime
from sort.sort import *
from util import get_car, read_license_plate, write_csv
from visualize import draw_border
import time


# load video
def carplateDetection(source, car_dt_model = 'yolov8n.pt', license_plate_dt_model = 'models/best.pt'):
    results = {}
    coco_model = YOLO(car_dt_model)
    license_plate_detector = YOLO(license_plate_dt_model)
    mot_tracker = Sort()
    cap = cv2.VideoCapture(source)
    start_time = time.time()
    frame_count = 0
    current_date = datetime.datetime.now()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 15
    out = cv2.VideoWriter(f'unprocessed_videos/{current_date}.mp4', fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    vehicles = [2, 3, 5, 7]

    # read frames
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        frame_count +=1
        ret, frame = cap.read()

        if frame_count > 5:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            print(f"Inference speed: {fps:.2f} FPS")
            frame_count = 0
            start_time = time.time()

        out.set(cv2.CAP_PROP_FPS, fps)
        print(fps)
        out.write(frame)

        if ret:
            results[frame_nmr] = {}
            # detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            # track vehicles
            if detections_ != []:
                track_ids = mot_tracker.update(np.asarray(detections_))


            # detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # assign license plate to car
                try: # because sometimes plates without cars appear and crash the programme
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)


                    if car_id != -1:

                        # crop license plate
                        license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                        # process license plate
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                        license_plate_crop_contrast = cv2.convertScaleAbs(license_plate_crop_gray, alpha=1.5)

                        # read license plate number
                        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_contrast)
                        if license_plate_text:
                            print(license_plate_text, license_plate_text_score)

                        if license_plate_text is not None:
                            results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                          'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                            'text': license_plate_text,
                                                                            'bbox_score': score,
                                                                            'text_score': license_plate_text_score}}

                            draw_border(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0),
                                        int(0.02 * (xcar2 - xcar1)),
                                        line_length_x=int(0.2 * (xcar2 - xcar1)),
                                        line_length_y=int(0.2 * (ycar2 - ycar1)))
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255),
                                          int(0.01 * (xcar2 - xcar1)))
                except:
                    pass
            cv2.imshow("DetectedObjects", frame)
            cv2.waitKey(10)


        # write results
    write_csv(results, f'{current_date}.csv')



