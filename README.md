# Project 3: Facial Expression Classification

## Ever wonder how powerful Machine Learning is?
![alt text](https://raw.githubusercontent.com/Donthave1/goodgamegang/master/images/fmh.png)


By **Team GoodGameGang**  
Edward Chen, Josh Shigaki, Peter Liu, Thomas Nakamoto


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

![alt text](https://raw.githubusercontent.com/Donthave1/goodgamegang/master/images/relight.png)


## 3 The Pre-Trained Deep Learning Nerual Network Model
Only using single layer of model to predict something complicated as human facial expression will be impossible to get any accurate result...  
Deep learning is a most popular/accurate technique to apply in computer vision, and it is capable to solve more complex problem like the one we have here. So we found a convolutional neural network (CNN) layered train model as our fundation and to build upon. CNNs work like human brain synapes works when processing/analyzing information.  
A typical architecture of a convolutional neural network contain an input layer, multiple convolutional layers, couple fully-connected dense layers, and one output layer.  
![alt text](https://raw.githubusercontent.com/Donthave1/goodgamegang/master/images/CNNs.png)  



## 4 Step of Validation
1. Teach machine to understand what a human face is:
Import pretrained model from Casscade Classifier (Haar Cascade)

2. Teach machine how to classify a emotion:
A Pretrained emotion classifier with dataset provided by Kaggle. The classifier reads gray scale image with 48 by 48 pixel resolution.

3. How to find/detect faces via computer vision: 
Using pretrained model with cascade classifier.detectMultiScale, it read the entire image, and capture pixels that has the human faces components. It crops the bottom left Corner and the width and height of the detected object.

4. How to identify the area to perform analysis and transform the data:
After capturing the face, we changed the image to gray scale, so we can compare the image at a normalized environment. The image must also be resize from your high resolution camera to 48 pixel by 48 pixel. The image information must also be digitalized so the the model could understand the image. First adding an array (Portfolio) to the pixels so the machine knows there's one image of 48 by 48 passing in. Then reshape with axis=-1 to tell the machine, each pixel has only 1 component, which is range by 0-255 for gray scale.

The pixel is also normalized by dividing 255 so machine can compare facial features without noise.

5. Predict (perform machine learning):
By inputing the (1,48,48,1) array into the Keras model, we are capable of computing the likelihood of our subject's emotion.

6. Return result transform into visual:
For image analysis, return the emotion with highest likelihood.

For video analysis, store all highest likelihood emotion in the database and graph all identified emotion as a pie chart

## 5 The Apps
 
Hosting on Local machine using Flask with video application as a separate app.
The separate application updates a local database that will provide information for graphing on Flask.


## 6 About the Team

**Team GoodGameGang**  
We are group of Data Analytic Bootcamp students that are passionate about using the power of machine learning to solve the real world challenges. 

## 7 References

1. [*"Dataset: Facial Emotion Recognition (FER2013)"*](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) ICML 2013 Workshop in Challenges in Representation Learning, June 21 in Atlanta, GA.

2. [*"Pre-trained Model: fer2013_mini_XCEPTION.119-0.65.hdf5"*](https://github.com/oarriaga/face_classification) by Face classification and detection from the B-IT-BOTS robotics team.

3. [*"Published Paper: Real-time Convolutional Neural Networks for Emotion and Gender Classification"*](https://github.com/oarriaga/face_classification/blob/master/report.pdf) by Face classification and detection from the B-IT-BOTS robotics team.

4. [*"OpenCV facial detection pre-train model: Haar Cascades"*](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html) by OpenCV documentation

## 8 Tool Sets
**Python:** (Library: TensorFlow, OpenCV, Pandas, Numpy, Flask, SQLAlchemy)  
**JavaScript:** (Library: Bootstrap, Gulp, nmp)   
**Database:** AWS  
**HTML/CSS** 
	
