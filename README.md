
# Game Sense

This project aims to process and analyze tennis matches by detecting players,ball and court. The primary goal is generate real time analysis based on player movements and ball tracking and event detections.


## Run Locally

This project requires python 3.10


Clone the project

```bash
  git clone https://github.com/AyanGairola/mM.git
```
or
```bash
  git clone git@github.com:AyanGairola/mM.git
```




Create a new Environment

using Anaconda

```bash
conda create gameSense
```

```bash
conda activate gameSense
```

OR

using Venv

```bash
python -m venv myenv
```

```bash
source myenv/bin/activate
```

Install Dependencies
```bash
pip install -r requirements.txt
```

Go to the project directory

```bash
  cd tennis
```

Go to Models directory

```bash
  cd models
```
Download the pretrained Models from the drive link
https://drive.google.com/drive/folders/1bs47xAaW56BFVQJx6tOJ45rTH0mq4evB?usp=sharing

Add the keypoints_model.pth to the models directory

Go to player_and_ball_detection directory

```bash
  cd player_and_ball_detection directory
```

Add the player and ball detection / best.pt from the drive link to the player_and_ball_detection directory

Go back a few directories

```bash
  cd ../..
```

You should be now in Tennis direcotry 

Open the file in your favourite code eidtor

```bash
  code .
```

Open main.py

On line no. 109 add your Gemini Api Key 

```python
api_key = "YOUR_GEMINI_API_KEY"  
```

If you don't have one no issues just leave the code as it is it will fallback to non generative commentary

```python
api_key = ""  
```

Processing the video will take a lot of time because of all the api calling to gemini and the processing so it's better to grab a bite at this point. 

We have provided the pickle file for the input video provided that will decrease the time by a lot. If not using the earlier provided input file. Open main.py 

On line 86 toggle the read_from_stub value to false and the stub_path to none. 

```python
detections = unified_tracker.detect_frames(video_frames, read_from_stub=False, stub_path=None)
```

Run main.py file with the input video's pathname as argument

```bash
python3 main.py input_vods/input_video1.mp4 
```

Output Video can be found out in output_vods direcotry


## Project Overview: Development of an ML-powered system for real-time analysis of tennis matches.

### Player and Ball Tracking 
Trained a yolov5 model on the dataset available at roboflow(https://universe.roboflow.com/jasiu-wachowiak-kqpcy/tennis-iu371). To make the ball tracking more robust we use interpolation for the points in which the ball has not been detected
We estimate the player’s position based on historical data or fallback to a default position if necessary. This ensures that the system can still function even when positional data is incomplete for some frames. 

 ### Court Detection and Event Analysis
 We use a pretrained resnet50 model available on the internet that returns the x,y postion of 14 keypoints across the court in a list, applied homography transformations for court boundary refinement and applied event detection techniques to detects in game events like ball hit and type of shots. 

### Commentary Generation
Integrate event-based dynamic generative ( Google Gemini) commentary engine that generates contextual insights based on in-game events.


### Mini Court Generation
Developed a Mini court based on actual court for better visualisations and calculations. 

### Player Stats
Generated player stats based on movement and ball shots. 

## Outputs
You can check out output videos in the output_videos folder

Some of the screenshots are
<img width="1512" alt="Screenshot 2024-09-12 at 3 01 44 AM" src="https://github.com/user-attachments/assets/ad35132a-5d89-413b-a993-96d105ccf42e">

<img width="1512" alt="Screenshot 2024-09-12 at 3 00 54 AM" src="https://github.com/user-attachments/assets/74713c4f-f99e-4c8a-ac9f-3499b8dd63a2">



## Known Issues

### Lack of Labelled Data 

Lack of appropriate labelled data

### Clay Court

The pretrained court detection model does not work well with clay courts

### Undertrained and Computationally Expensive Models

Due to high computationan cost of training the models, the models have been trained on less epochs(60) which may cause issue.

### Bounce detection

Due to lack of different camera angles, bounce detection is not so accurate.

### Rally Counter

Fails to reset after a point

### Score Calculation

Due to inconsistenices in bounce detection and interpolation, there is noise in the score detection

## Future Work

### Pose Estimation 

Leverage advanced pose estimation models for accurate and advanced shot detection

### Doubles

Expand the project for doubles game

### Coaching Assistance based on advanced analytics

Incorporating advanced analytics can significantly improve a player’s understanding of their strengths and weaknesses also helping them with training programs accordingly.

### Match outcome prediction
The goal is to leverage player performance data, match conditions, and situational analysis to accurately predict who will win a match.

### Text to Speech

Utizile advanced tts models to convert our generated commentary into voiceover

### Support for Mutiple Camera Angles

Add support for mutiple camera angles to imporve user experience

## Refrences and Resources

https://universe.roboflow.com/

https://medium.com/@kosolapov.aetp/tennis-analysis-using-deep-learning-and-machine-learning-a5a74db7e2ee


To generate commentary - Google gemini

