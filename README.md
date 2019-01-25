# Project 3: Facial Expression Classification

## Ever wonder how powerful Machine Learning is?
![alt text](https://raw.githubusercontent.com/Donthave1/goodgamegang/master/images/fmh.png)

# [Site]()

By **Team GoodGameGang**  
Edward Chen, Josh Shigaki, Peter Liu, Thomas Nakamoto
  
## The Real World Challenge: 
Have you ever got into trouble because you can’t read emotion? Your significant other is feeling uncomfortable and sick when you continue to make fun of her; you’re telling an inappropriate joke next to your parents on grandma’s birthday celebration; you’re spilling out the deepest secret of your best friend in front of his significant other. I don’t know how you are still here alive, but for sure you had enough frustration in life. That is the reason you come to us! We built a emotion recognizer to read facial emotion spontaneously. You will no longer need to suffer from the punishment of not catching someone’s facial gesture and hints anymore!

Using a web application hosted on Amazon Web Service to intake user image through a webcam. By passing in the image pixel to backend analyzer our pre-trained model is capable to detect user facial features. Furthermore, with the facial features recognized, our emotion recognition model will be able to analyze users’ emotion and respond with one of the seven different states: angry, disgust, fear, happy, sad, surprise, neutral.


## Table of Contents
1. [Motivation](#1-motivation)
2. [The Database](#2-the-database)
3. [The Pre-Trained Model](#3-the-model)
4. [Model Validation](#4-step-of-validation)
5. [The Apps](#5-the-apps)
6. [About Us](#6-about-the-team)
7. [References](#7-references)
8. [Tool Sets](#8-tool-sets)


## 1 Motivation
Human can easily tell apart how other person feel based on how their facial expression is at that moment. Human facial emotion are expressed through the combination of specific sets of facial muscles. Expression can be seen straigt forward even a baby can recognize, yet at the same time it is complex enough that a smile might contain vast amount of information with the state of mind that person has.  
Human brain was born and trained to recognize pattern accuratly and fast enough even the best computer chip cannot compete. We are so use to reading and seen people faces that it become natural of just predict what other is thinking or feeling. But can it be done through computer vision? is it possible to have a computer to guess what the person is feeling at the time with just their facial expression?  
Human facial expressions can be easily classified into 7 basic emotions: happy, sad, surprise, fear, anger, disgust, and neutral. But for computer, it would not know anything about happy or sad or what does it means beside the textual definition (with code pre-programed).  

So today, our team took up this challenge and we are planning to answer it with deep learning neural network that give computer vision the ability to recognize how one feel at the moment. 


## 2 The Database
The dataset that our model was trained on is from a [Kaggle Facial Expression Recognition Challenge (FER2013)](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge), which comprises a total of 35887 pre-cropped, 48-by-48-pixel grayscale images of faces each labeled with one of the 7 emotion classes: anger, disgust, fear, happiness, sadness, surprise, and neutral.

![alt text]()


## 3 The Pre-Trained Deep Learning Nerual Network Model
Only using single layer of model to predict something complicated as human facial expression will be impossible to get any accurate result...  
Deep learning is a most popular/accurate technique to apply in computer vision, and it is capable to solve more complex problem like the one we have here. So we found a convolutional neural network (CNN) layered train model as our fundation and to build upon. CNNs work like human brain synapes works when processing/analyzing information.  
A typical architecture of a convolutional neural network contain an input layer, multiple convolutional layers, couple fully-connected dense layers, and one output layer.  
![alt text]()  




## 4 Step of Validation


## 5 The Apps



## 6 About the Team

**Team GoodGameGang**  
We are group of Data Analytic Bootcamp students that are passionate about using the power of machine learning to solve the real world challenges. 

## 7 References

1. [*"Dataset: Facial Emotion Recognition (FER2013)"*](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) ICML 2013 Workshop in Challenges in Representation Learning, June 21 in Atlanta, GA.

2. [*"Pre-trained Model: fer2013_mini_XCEPTION.119-0.65.hdf5"*](https://github.com/oarriaga/face_classification) by Face classification and detection from the B-IT-BOTS robotics team.

3. [*"Published Paper: Real-time Convolutional Neural Networks for Emotion and Gender Classification"*](https://github.com/oarriaga/face_classification/blob/master/report.pdf) by Face classification and detection from the B-IT-BOTS robotics team.

4. [*"OpenCV facial detection pre-train model: Haar Cascades"*](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html) by OpenCV documentation

## 8 Tool Set
**Python:** (Library: TensorFlow, OpenCV, Pandas, Numpy, Flask, SQLAlchemy)  
**JavaScript:** (Library: Bootstrap, Gulp, nmp)   
**Database:** AWS  
**HTML/CSS** 
	
