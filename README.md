# Colorizer
AI based Image Coloring of B/W images

This project is based on a refrence from a research paper titled 'Colourful Image Colorization' authored by 

## Project Directory
The directory includes app.py as the main flask server application. It misses a directory named model which stores the binary caffe model for prediction. The moodel is linked at __ due to file size limit on Github. 

## Requirements
The project needs to be installed with following requirements

- python 3
- numpy
- pillow
- open-cv-contrib python
- imutils
  done. You are good to go.
 
## Working 
AI based image recoloring of black and white images uses deep neural networks trained over a large sample of images (1.5 Million) in LAB color space with L channel as input and ab channel as output.
Here is a video demonstrating recoloring of an Iconic Hindi Bollywood B&W song https://www.linkedin.com/posts/piyush-verma-258800163_ai-deeplearning-webapp-activity-6633071583818805248-ET6-.

### For Example 
![Alt desc](http://colorizer.pythonanywhere.com/static/images/example.jpg)
