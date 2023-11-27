## Task- Object Detection with YOLO

- **Data Preparation:**
Given approximately 167 Koala Images and instructions to label those images and build a model on YOLO. I labeled images with the open-source online annotator LabelMe. Create a bounding box for every image and have 5 values one each such as label, x_min, y_min, x_max, y_max.
After annotating all the images upload on Google Colab for further investigation. Import every JSON file and append those together to create a Txt file for each. Each txt file will contain 5 values and we know, but we need different values like x_mid, y_mid, width, and height.
Calculate and process that Txt files, and finally create Txt files with expected values.
Now match every image with a label to find the right image for processed images. Having 163 labels from 167 images discard 4 images and map those images with corresponding images. Now divide images with train and test and on train images use fold cross-validation with
n_splits=3. Create a directory and save processed images, labels, and classes.

- **YoLo Format Creation:**  To train the YOLO model, create a YAML file that contains the train, valid, number of classes, and classes. For validation 1 folds and the rest of the two folds are used as training data.

- **Import YOLO:** You only look Once is the full form of YOLO. For this dataset, I used the yolov5 model with small weights. Clone the model with the current directory and install the requirements.txt. After checking the installation and running the pre-define
  detect the image. Now time to train the model, FYI I trained the model three times using different fold and image sizes. Universally I used batch size=16, epochs=50, images_size=220 one time and two times 640. Not done any Hyperparameter tuning which is a necessary point
  to train any pre-train model. If did that then the box loss would be decreased besides the evaluation metrics would be increased. But on top of that got better results on difference fold.

|Fold| box_Loss|Precision|Recall |mAP|
|-----| ----- | ----- |-----|---------|
|1|0.0209| 0.963|0.978|0.993|
|1|0.02092| 0.982|0.993|0.995|
|3|0.0199| 0.992|0.993|0.995|


- **Create An API:** I created an API for better usability of the model with FastAPI. where set an endpoint to predict the bounding box of the image which I want to select.

 **END NOTE**I learned a lot of things while doing this task, The preprocessing steps were interesting, and also selected a model for small images, We all know Yolo works better on high-resolution images like 640x640 or 1280x1280. Find out the perfect model and parameter is more
  interesting part while training a model like batch_size, optimizer, which weights better, and so on. Faced different types of challenges while creating API Locally with FastAPI finally solved those issues.

  
  
