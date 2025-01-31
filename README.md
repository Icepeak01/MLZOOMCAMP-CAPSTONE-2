# MLZOOMCAMP-CAPSTONE-2

# Pneumonia-Detection-CNN

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Description](#problem-description)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Contact](#contact)

## Project Overview

This project aims to develop a machine learning model to predict loan repayment status based on various applicant features. Leveraging multiple classification algorithms, including Logistic Regression, Random Forest, XGBoost, and Neural Networks, the model provides accurate predictions to assist financial institutions in making informed lending decisions. The application is deployed using Flask and Docker, making it accessible via a web interface for real-time predictions.

## Problem Description

Loan default is a critical issue for financial institutions, leading to significant financial losses and impacting credit availability. Accurately predicting whether an applicant will repay a loan enables lenders to mitigate risks, tailor loan offerings, and ensure financial stability. This project leverages data science techniques to build a predictive model that assesses the likelihood of loan repayment based on applicant demographics, financial history, and other relevant features.

### How the Model Helps:

- Risk Mitigation: Identifies high-risk applicants, reducing potential defaults.
- Operational Efficiency: Streamlines the loan approval process by automating risk assessment.
- Personalized Offerings: Enables lenders to customize loan terms based on applicant profiles.
- Financial Stability: Enhances the overall financial health of lending institutions by minimizing losses.



## Dataset

The dataset used for this project is sourced from [Kaggle's Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

### Dataset Details:
- Features:

  - gender: Gender of the applicant (M, F)
  - type_of_residence: Type of residence (Rented Apartment, Own House, Parents Apartment)
  - educational_attainment: Highest education level achieved
  - employment_status: Employment status (Employed, Self Employed, Others)
  - sector_of_employment: Sector where the applicant is employed
  - requested_amount: Amount of loan requested
  - purpose: Purpose of the loan (Business, Medical, Others, Personal)
  - loan_request_day: Day of the week the loan was requested
  - age: Age of the applicant
  - selfie_id_check: Status of selfie ID verification (Successful, Others)
  - loans: Number of existing loans
  - phone_numbers: Number of phone numbers associated with the applicant
  - mobile_os: Operating system of the applicant's mobile device
  - income_range: Income bracket of the applicant
    
- Target:
  - target: Loan repayment status (1 for Paid, 0 for Not Paid)

### Accessing the Data:
- Committed in Repository: The data/ directory contains the Lendsqr_Data_Science_Assessment_Dataset_v1.csv file.
- Alternatively: If the dataset is too large to commit, follow the download instructions below.

## Project Structure

loan-prediction-project/
├── Dockerfile
├── README.md
├── flaskapp.py
├── requirements.txt
├── ldsqr_lr_model.pkl
├── ldsqr_scaler.pkl
├── col_name.pkl
├── templates/
│   ├── home.html
│   └── prediction.html
├── static/
│   └── styles.css
├── notebook.ipynb
└── images/
    └── confusion_matrix.png


## Installation

### Prerequisites

- **Docker:** Ensure Docker is installed on your machine. [Installation Guide](https://docs.docker.com/get-docker/)
- **Python 3.9:** Required for running training and prediction scripts.

### Clone the Repository

```bash
git clone https://github.com/your-username/loan-prediction.git
cd loan-prediction
```

### Setup Python Environment
It's recommended to use a virtual environment
```bash
# Using virtualenv
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r app/requirements.txt
```



### Usage
Running the Flask Application Locally
```bash
# Activate Virtual Environment
source venv/bin/activate

# Navigate to the app Directory
cd app

# Run the Application
python flaskapp.py
```

### Docker Deployment
```bash
# Build the Docker Image:
docker build -t loan-prediction-app .

# Run the Docker Container:
docker run -d -p 5000:5000 loan-prediction-app

```
Access the App:

Open your browser and navigate to http://localhost:8080.

### Accessing the Deployed Application
The application is deployed on Render and can be accessed via the following link:

👉 [Loan prediction App](https://ml-zoocamp-project-1.onrender.com/)

## Training the Model
The training process involves data preparation, feature engineering, exploratory data analysis (EDA), model training with multiple classifiers, and evaluation. The final model is saved for deployment.

Running the Training Script
Ensure Dependencies are Installed:
``` bash
pip install -r app/requirements.txt
```
Run the Training Script:
``` bash
python scripts/train.py
```

Model Saving:

The trained model, scaler, and column names are saved in the app/ directory as:
  - ldsqr_lr_model.pkl
  - ldsqr_scaler.pkl
  - col_name.pkl

## Model Evaluation
After training, the model was evaluated on the test dataset with the following results:

Classification Report:
``` markdown
              precision    recall  f1-score   support

      NORMAL       0.96      0.70      0.81       234
   PNEUMONIA       0.84      0.98      0.91       390

    accuracy                           0.88       624
   macro avg       0.90      0.84      0.86       624
weighted avg       0.89      0.88      0.87       624
```

### Training Performance:
``` markdown
  Epoch	    Accuracy	    Loss	   Val Accuracy	    Val Loss	 Learning Rate
      1	     0.7111	    0.5754	   0.9406	        0.1885        1e-04
      2	     0.9127	    0.2273	   0.9386	        0.1545	      1e-04
      3	     0.9283	    0.1706	   0.9473	        0.1349	      1e-04
      4      0.9415	    0.1415	   0.9703	        0.0926	      1e-04
      5	     0.9523	    0.1205	   0.9645	        0.0998	      1e-04
      6	     0.9521	    0.1182	   0.9703	        0.0882	      1e-04
```
### Fine-Tuning Performance:
``` markdown
  Epoch	    Accuracy	  Loss	      Val Accuracy	   Val Loss	  Learning Rate
      6	      0.9050	  0.3118	    0.9664	      0.0953	      1e-06
      7	      0.9252	  0.2134	    0.9655	      0.1013	      1e-06
      8	      0.9128	  0.2366	    0.9597	      0.1065	      1e-06
```

## Deployment

The application is containerized using Docker and deployed to [Render](https://render.com/).

### Access the Deployed Application

👉 [Pneumonia Detection App](https://ml-zoocamp-project-1.onrender.com)


### Deployment Steps

1. **Build the Docker Image:**

    ```bash
    docker build -t pneumonia-detection-cnn .
    ```

2. **Run Locally for Testing:**

    ```bash
    docker run -d -p 8080:8080 pneumonia-detection-cnn
    ```

3. **Push to GitHub:**

    Ensure all changes, including the Dockerfile and model, are pushed to your GitHub repository.

4. **Deploy to Render:**

    - Connect your GitHub repository to Render.
    - Render will automatically build and deploy the Docker container.
    - Ensure the `PORT` environment variable is set correctly if needed.

## Technologies Used

- **Programming Languages:** Python
- **Frameworks & Libraries:**
  - Flask
  - TensorFlow & Keras
  - OpenCV
  - Matplotlib
  - Seaborn
  - Gunicorn
- **Tools:**
  - Docker
  - Git & GitHub
  - Render (for deployment)


## Deployment

![image](https://github.com/user-attachments/assets/9668f398-fe79-451d-8371-9778bfdb76a2)
*Screenshot of the Pneumonia Detection App while uploading an X-ray image.*


![image](https://github.com/user-attachments/assets/e4f110a6-c488-4e94-b357-fdcb50ec9532)

*Screenshot of the Pneumonia Detection App showing an uploaded X-ray image and the Grad-CAM heatmap.*

## Dataset

The dataset used in this project is sourced from [Kaggle's Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

### Download Instructions:

1. Create a Kaggle account if you don't have one.
2. Navigate to the [dataset page](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
3. Click on the "Download" button to download the dataset.





