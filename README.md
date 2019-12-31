# PotholeDetector
Download the Daataset from Kaggle 

This contains arount 330 images of potholes and roads

https://www.kaggle.com/atulyakumar98/pothole-detection-dataset

The dataset is strucctured in this format
--pothole-detection-dataset 
  --normal
  --potholes

From the 330 images 80% (arund 264 images) have been used for training and the rest for validating
Place the 264 images from normal and potholes under Pothole and Road in Train folder and the rest in Validation folder.

The dataset directory is to be structured in this Format

--PotholeDatset
--Train
  --Pothole
  --Road
--Validation
  --Pothole
  --Road
  
Running the file creates a "trained.h5" file.  
Feel free to tweak your own settings.

Running the "testpothole.py" by modifying the load_image function's parameter will print the statement "This is a" + the prediction.
