# AI Home Monitoring with Nvidia Jetson Nano 
An example development repository for using Nvidia Jetson Nano or Xavier as health monitor using computer vision. It show case the Open Pose, and Face Recognition, and Emotion Analysis (all GPU code) running in real-time on the Jetson Nano platform. 

Please read this medium post for more details about the setup. 

https://towardsdatascience.com/using-cv-and-ml-to-monitor-activity-while-working-from-home-f59e5302fe67

![Pose + Face](https://github.com/raymondlo84/nvidia-jetson-health-monitor/blob/master/sample_outputs/pose_face.gif)

# Dependencies

You have to install the following applications and libraries to run the code.

- Motion (IP Camera) - https://motion-project.github.io/index.html

- Tensorflow & TF Pose - https://github.com/karaage0703/jetson-nano-tools

- Face Recognition - https://github.com/ageitgey/face_recognition 

- Emotion Analysis - https://github.com/abhijeet3922/FaceEmotion_ID (for the model)

- OpenCV - https://github.com/karaage0703/jetson-nano-tools


# Usage 

To run the core demo, execute the following script in terminal 
```
$ cd nvidia-jetson-ai-monitor
$ python3 run_webcam_ip.py --model=mobilenet_thin --resize=320x160
```

# References
- https://medium.com/@ageitgey/build-a-hardware-based-face-recognition-system-for-150-with-the-nvidia-jetson-nano-and-python-a25cb8c891fd

- https://pythonspot.com/flask-and-great-looking-charts-using-chart-js/

- https://karaage.hatenadiary.jp/

- https://github.com/karaage0703/jetson-nano-tools

- https://toramamma.blogspot.com/2019/04/jetson-nano-tensorflowopenpose.html

- https://github.com/ageitgey/face_recognition
