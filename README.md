# real_time_carplate_detection
## General information
This project is a significantly modified version of [this library](https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8). 
It uses Yolov8 model alongside [sort object tracking library](https://github.com/abewley/sort) and [EasyOcr](https://github.com/JaidedAI/EasyOCR) for character recognition.
## Real time detection
The ```from main import carplateDetection``` function takes the link to a live video as an argument, detects carplates and visualizes targeted cars on an output video. Then text from the carplates processed by EasyOCR and written into a .csv file. Live video is recorded and saved into a ```unprocessed_videos``` folder. Weight for neural networks could also be changed via ```car_dt_model``` and ```license_plate_dt_model``` parameters. This function could also be used to process recorded videos or retrieve video from a connected device.
## Visualization
Having the file with .csv file with detections and the video it describes results could be visualized with ```from visualize import visualize``` function that takes path to .mp4 file as an argument.
## Example of visalization
![detected carplates](https://github.com/C1l1r/real_time_carplate_detection/blob/master/demo.gif)

## Instalation

```
docker-compose -f docker_compose_file_path up
```

#### Changes compared to the original library include but do not limit to:

* added compability with the livestreams
* optimized carplate image preprocessing
* original weights for carplate detection were [changed](https://github.com/MuhammadMoinFaisal/Automatic_Number_Plate_Detection_Recognition_YOLOv8/tree/main) which led to improved performance
* removed hardcoded dependencies to 4k video resolution
* fixed bug that would crash the application:
  - if the carplate was detected without car
  - if no detections were made thruout the whole video
  - if the fps changes throut the video
  - if the tracked object leaves the frame
* removed hardwared compability with british only carplates
* white text background now adapts to text size 
