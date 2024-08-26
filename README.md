# Age and Gender Detection Model

# Overview
This project implements an Age and Gender detection model using deep learning techniques. The model is designed to predict both the age and gender of a person from a single input image. It leverages the ResNet152V2 architecture as the backbone for feature extraction, with custom heads for the age and gender predictions.

# Model Architecture
Base Model
Backbone: ResNet152V2 (pre-trained on ImageNet)
Input Shape: (128x128x3) for Gender Detection and (224x224x3) for Age Detection.
Gender Detection Head
Global Average Pooling Layer: Reduces the feature map dimensions.
Dropout Layer: Prevents overfitting by randomly dropping some units.
Dense Layer: Outputs a single neuron with a sigmoid activation for binary classification (Male/Female).
Age Detection Head
Global Average Pooling Layer: Similar to the gender detection head.
Dropout Layer: Prevents overfitting.
Dense Layer: Outputs a single neuron with a linear activation for regression (age prediction).
# Dataset
The dataset used for training includes images of faces with labeled age and gender. The filenames follow the format: age_gender_...jpg. The age is extracted from the filename for the age prediction task.

Data Augmentation
To enhance the model's performance and generalization, data augmentation is applied:

Horizontal Flip: Randomly flips the image horizontally.
Rotation: Rotates the image within a specified range.
Zoom: Randomly zooms in on the image.
Training
The model is compiled using:

Optimizer: Adam with a learning rate of 1e-5.
Loss Function:
Gender: Binary Crossentropy for binary classification.
Age: Mean Absolute Error (MAE) for regression.
Early Stopping is used to prevent overfitting, halting training when the validation loss no longer improves.

How to Run
1. Clone the Repository

git clone https://github.com/abhinavyadav11/age-genderDetection.git
cd age-gender-detection

2. Install Dependencies
Ensure you have Python 3.8+ installed. Then install the required packages:

pip install -r requirements.txt

3. Prepare Dataset
Place your dataset in the data/ directory. The dataset should be structured as follows:

data/
  |- train/
      |- age_gender_1.jpg
      |- age_gender_2.jpg
      ...
  |- validation/
      |- age_gender_3.jpg
      |- age_gender_4.jpg
      ...

4. Load the Model

a) Gender Model : https://www.dropbox.com/scl/fi/bv6mddeixhqcqm66em7uu/Gender_Prediction_model.h5rlkey=nte1f7e3jafgi1fxghxmf4g4f&st=xq68w5te&dl=0

b) Age Model : 
https://www.dropbox.com/scl/fi/lhehid0iw9g315805od5a/Age_Prediction_model.h5?rlkey=7y6wc5jxe6mqiz1hjewehue4d&st=28m6jez8&dl=0




5. Inference
To predict the age and gender for a new image:


python predict.py --image_path path/to/image.jpg
Results
The model achieves the following performance:

Gender Prediction Accuracy: 92%
Age Prediction MAE: <MAE value>
Model Saving and Loading
The model is saved in HDF5 format after training. You can load the saved model using:


from keras.models import load_model
model = load_model('age_gender_model.h5')
# Future Work
Model Optimization: Explore other architectures and hyperparameters.
Expand Dataset: Include more diverse and large-scale datasets.
Real-time Prediction: Implement the model in a real-time application using OpenCV.
# Acknowledgments
The ResNet152V2 model pre-trained on ImageNet.
The datasets used for training and validation.
# License
This project is licensed under the MIT License - see the LICENSE file for details.

Contributors:

Abhinav Yadav

For more details or issues, please reach out to #abhijust36@gmail.com.
