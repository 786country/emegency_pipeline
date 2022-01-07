# Disaster Response Pipeline Project

Location: https://github.com/786country/emegency_pipeline
## 1. Installation 

The anaconda distribution of python will cover most packages, however the following package(s) will be required to be installed: 

`missingno`
`plotly`
`flask`

This can be installed in anaconda using the following commands: 

`conda install -c anaconda nltk`
`conda install -c plotly plotly`
`conda install -c anaconda flask`

## 2. Project Motivation 

This project was completed as part of the Data Science Nanodegree Program at Udacity. 

## 3. File Descriptions 

The Repo Folders are listed in the below table. (ignoring repository standard files)

| Folder Name | Description                                                          |
|-------------|----------------------------------------------------------------------|
| app         | Folder containing web app scripts                                    |
| data        | Folder containing raw data and processing scripts                    |
| models      | Folder containing models and training scripts                        |
| Notebooks   | Folder containing Notebook used to perform data engineering/science  |

The files are the following: 

| Folder Name                              | Description                                              |
|------------------------------------------|----------------------------------------------------------|
| data\process_data.py                     | Script used to process data                              |
| models\train_classifier.py               | Script used to train model                               |
| app\run.py                               | Script used to run web dashboard                         |
| Notebooks\ETL Pipeline Preparation.ipynb | Notebook used to prepare data\process_data.py            |
| Notebooks\ML Pipeline Preparation.ipynb  | Notebook used to prepare models\train_classifier.py      |
| data\disaster_messages.csv               | Data containing the disaster messages                    |
| data\disaster_categories.csv             | Data containing the disaster categories                  |
| data\DisasterResponse.db                 | Data containing the cleaned merged dataset for modelling |

## 4. Project Challenges 

The main challenge with the dataset was the training time, the datasets large size coupled with the multiple models required (36). This was especially hard for grid search as the different permutations of models made the training time quite long. 

## 5.Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


