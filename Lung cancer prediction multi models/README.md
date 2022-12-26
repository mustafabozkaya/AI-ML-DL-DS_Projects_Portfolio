# Lung Cancer Prediction Multi Models

## Table of Contents

- [Lung Cancer Prediction Multi Models](#lung-cancer-prediction-multi-models)
  - [Table of Contents](#table-of-contents)
  - [About the Project](#about-the-project)
  - [About Dataset](#about-dataset)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Examples](#examples)
  - [Additional Information](#additional-information)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## About the Project

This project is a machine learning model that uses a dataset to predict whether a person has lung cancer or not. The model is trained using a variety of features, including age, smoking status, and medical history. The project includes several models (e.g. linear SVC, random forest, and XGBoost) and allows users to compare their performance and feature importance.

## Project Structure

The project has the following structure:

├── data/
│   └── data.csv
├── models/
│   ├── model1.pkl
│   ├── model2.pkl
│   └── model3.pkl
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   └── model_training_and_evaluation.ipynb
├── presentations/
│   └── project_presentation.ppt
├── src/
│   ├── data_processing.py
│   ├── main.py
│   ├── model_training.py
│   └── prediction.py
├── .gitignore
├── README.md
└── requirements.txt

The **data** folder contains the raw and processed data used in the project.

The **models** folder contains the trained machine learning models.

The **notebooks** folder contains Jupyter notebooks for exploratory data analysis and model training and evaluation.

The **presentations** folder contains presentation materials for the project.

The **src** folder contains the source code for the project, including scripts for data processing, model training, and prediction.

The .**gitignore** file lists files and directories that should be ignored by Git.

The **README.md** file contains information about the project, including its purpose and main features, setup instructions, usage instructions, and examples.

The **requirements.txt** file lists the dependencies for the project.

## About Dataset

The effectiveness of cancer prediction system helps the people to know their cancer risk with low cost and it also helps the people to take the appropriate decision based on their cancer risk status. The data is collected from the website online lung cancer prediction system .
Total no. of attributes:16
No .of instances:284
Attribute information:

Gender: M(male), F(female)
Age: Age of the patient
Smoking: YES=2 , NO=1.
Yellow fingers: YES=2 , NO=1.
Anxiety: YES=2 , NO=1.
Peer_pressure: YES=2 , NO=1.
Chronic Disease: YES=2 , NO=1.
Fatigue: YES=2 , NO=1.
Allergy: YES=2 , NO=1.
Wheezing: YES=2 , NO=1.
Alcohol: YES=2 , NO=1.
Coughing: YES=2 , NO=1.
Shortness of Breath: YES=2 , NO=1.
Swallowing Difficulty: YES=2 , NO=1.
Chest pain: YES=2 , NO=1.
Lung Cancer: YES , NO.

## Setup

To set up the project, you will need to install the following dependencies:

- Python 3.7 or later
- NumPy
- Pandas
- Scikit-learn
- Lazypredict

To install the dependencies, you can use a package manager such as `pip`:

pip install -r requirements.txt

1. Make sure you have the necessary dependencies installed. You can do this by running `pip install -r requirements.txt`.
2. Navigate to the `src` directory and open the file `main.py`.
3. Set the `data_path` argument to the path of the `data.csv` file in the `data` directory.
4. Run the `main.py` file using the command `python main.py`. This will train the models and save them to the `models` directory.
5. To make predictions using the trained models, you can use the `predict()` function in the `prediction.py` file. This function takes as input the path to the model file and the data to be predicted on, and returns the prediction. For example:

<pre><div class="bg-black mb-4 rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre-wrap hljs language-lua">from prediction import predict

model_path = 'models/model1.pkl'
data = [[62, 'M', 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]]  # example data
prediction = predict(model_path, data)
print(prediction)
</code></div></div></pre>

## Examples

Here are some examples of how you can use the trained models and the `predict()` function:

1. To use the first model to make a prediction on a single instance of data:

<pre><div class="bg-black mb-4 rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre-wrap hljs language-lua">from prediction import predict

model_path = 'models/model1.pkl'
data = [[62, 'M', 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]]  # example data
prediction = predict(model_path, data)
print(prediction)
</code></div></div></pre>

2. To use all three models to make predictions on a dataset:

<pre><div class="bg-black mb-4 rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre-wrap hljs language-scss">from prediction import predict

model_paths = ['models/model1.pkl', 'models/model2.pkl', 'models/model3.pkl']
data = [[62, 'M', 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1], 
        [62, 'M', 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1], 
        [62, 'M', 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]]  # example data
predictions = [predict(model_path, data) for model_path in model_paths]
print(predictions)
</code></div></div></pre>

## Additional Information

* You can find more information about the models and their performance in the `notebooks/model_training_and_evaluation.ipynb` notebook.
* You can find more information about the data and its features in the `notebooks/exploratory_data_analysis.ipynb` notebook.

## Contributing

If you would like to contribute to this project, please follow these guidelines:

* Fork the repository and create a new branch for your changes.
* Make sure your code is well-documented and follows PEP 8 style guidelines.
* Test your code thoroughly before submitting a pull request.

## Usage

To use the project, you can run the following command:

python main.py --data_path=`<path to data.csv>`

Copy code

The `--data_path` argument is required and should be replaced with the path to the `data.csv` file.

The program will train and evaluate the models on the dataset, and then display the accuracy scores and feature importances for each model.

## Examples

Here is an example of how to use the project to predict lung cancer:

```python
from main import predict

predictions = predict(data)
print(predictions)

# Output:
# [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
```

The predict function takes a DataFrame as input and returns a list of predictions (either 0 or 1) for each sample in the DataFrame.

## Additional Information

The project uses cross-validation to evaluate the models. The number of folds can be changed by modifying the `NUM_FOLDS` variable in `main.py`.

The project uses the following models:

- Linear SVC
- Random Forest
- XGBoost
- K Nearest Neighbors
- Decision Tree
- Extra Trees Classifier
- Ada Boost Classifier
- Gradient Boosting Classifier
- Gaussian Naive Bayes
- Quadratic Discriminant Analysis
- Logistic Regression
- Ridge Classifier
- Passive Aggressive Classifier
- SGD Classifier
- Bernoulli Naive Bayes
- Multinomial Naive Bayes
- Linear Discriminant Analysis
- Nearest Centroid
- Bagging Classifier
- Calibrated Classifier CV
- Linear SVC
- NuSVC

## Contributing

If you would like to contribute to the project, you can fork the repository and submit a pull request. If you find any bugs, please submit an issue.

## License

This project is licensed under the terms of the MIT license.

## Acknowledgements

Here is a list of sources where you can find datasets about lung cancer, along with instructions on how to access and download the data:

* The National Cancer Institute's Surveillance, Epidemiology, and End Results (SEER) Program: This program collects cancer incidence and survival data from population-based cancer registries covering approximately 34% of the US population. You can access data on lung cancer and other types of cancer through the SEER Cancer Statistics Review or the SEER Research Data (1973-2017) on the SEER website (https://seer.cancer.gov/). To download a dataset as a CSV file, follow the instructions on the SEER website to generate the dataset, and then click the "Export Data" button. Alternatively, you can use the SEER*API to access SEER data programmatically.
* The Cancer Genome Atlas (TCGA): This is a comprehensive and coordinated effort to accelerate our understanding of the molecular basis of cancer through the application of genome analysis technologies, including large-scale genome sequencing. You can access data on lung cancer and other types of cancer through the TCGA Data Portal on the National Cancer Institute website (https://www.cancer.gov/tcga).
* The Cancer Data Science Network (CDSN): This is a network of cancer data science experts and resources that provides access to data, tools, and services to support cancer research. You can access data on lung cancer and other types of cancer through the CDSN Data Portal on the National Cancer Institute website (https://cdsn.nci.nih.gov/).
* The International Association for the Study of Lung Cancer (IASLC): This is a professional organization dedicated to the study of lung cancer and other thoracic malignancies. You can access data on lung cancer and other types of cancer through the IASLC Data Center on the IASLC website (https://www.iaslc.org/iaslc-data-center).
* Publicly available datasets on online repositories such as Kaggle or the UCI Machine Learning Repository: These repositories host a wide range of datasets on various topics, including lung cancer. You can search for lung cancer datasets on these websites (https://www.kaggle.com/, https://archive.ics.uci.edu/ml/index.php) and download them for use in your research or analysis.
* The Google Dataset Search: This is a search engine for datasets that helps you find datasets from across the web.Go to the Google Dataset Search website (https://datasetsearch.research.google.com/) and enter a query (e.g. "Lung Cancer") in the search bar.

Browse through the search results and find a dataset that you are interested in.

Click on the dataset to view more details and access the data.

Depending on the dataset, you may be able to download the data as a CSV file or access it through an API or web service. You can find more information on how to access the data in the dataset's documentation or by contacting the dataset provider.

- [Lung Cancer dataset](https://archive.ics.uci.edu/ml/datasets/Lung+Cancer)
- [Lung cancer dataset](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)
- [Lung Cancer dataset](https://archive.ics.uci.edu/ml/datasets/Lung+Cancer)
