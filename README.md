# CapturingSmiles: Automatic Smile Detector
Praneeth Prathi, Andrew Lin

# Data Processing
We utilized the [GENKI-4K](https://inc.ucsd.edu/mplab/398/) dataset of images containing both labeled smiling and non-smiling faces. We utilized OpenCV's free library to access pretrained HaarCascade models for frontal face detection of each image from our original dataset. We converted each image into greyscale and utilized OpenCV functions to read the images into matrix pixel representations. We then converted the data to fit the size of the feature inputs to each of our developed models. The dataset containing the cropped images can be found on Google Drive [here](https://drive.google.com/drive/folders/1pQUtIhwlqTHCXRtjHlm_r_anNdIrg7Db?usp=sharing), as the data size was too large to upload. To run the notebook, the data folder needs to be downloaded to same directory as the models notebook.

Models:
- RandomForestClassifier Ensemble
- SupportVectorClassifier
- MultilayerPerceptronClassifier
- Simple CNN
- LeNet Architecture CNN
- AlexNet Architecture CNN

For each model with necessary hyperparameters, we performed GridSearchCV to select the optimal hyperparameters based on the scores of the validation set. Since our dataset consists of only 4000 images, we evaluated model results by performing k-fold cross validation with 10 folds. 

The ExploreData notebook contains exploratory data analysis steps, and the models notebook contains all of the training, outputs, and metric evaluation for our models. 

# Live Camera Application
The videocapture.py file uses the AlexNet trained model and OpenCVs VideoStream to read frames from the camera and predict whether you are smiling or not. If the prediction is positive for 10 consecutive frames, your image will be saved to the results/ directory. To run this application, the pretrained model needs to be downloaded from [here](https://drive.google.com/file/d/1PgAPne9qyFekXQRR3nPV_aM297hB-Oeb/view?usp=sharing), and unzipped in the same directory as videocapture.py. Then, the python file needs to be run (any necessary dependencies will need to be installed). 
