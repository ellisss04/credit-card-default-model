# Credit Card Default Prediction using Deep Learning

This project is part of an assignment aimed at evaluating students' proficiency in building, implementing, validating, and evaluating deep learning models for credit card default prediction using a provided dataset. The assignment also assesses their ability to identify key challenges within the dataset, propose solutions, and effectively communicate their findings and results.
Objective

The primary objective of this project is to develop and evaluate a deep learning model for predicting credit card defaults based on the provided CCD (credit card default) dataset. The project involves several key tasks including model design, validation, evaluation, and reporting.
Tasks

Design a deep learning technique for predicting credit card defaults from the CCD dataset.
Validate and evaluate the performance of the model.
Write a short report (maximum 2000 words) describing the architecture of the model, the training process, main deep learning challenges encountered, findings, and attempts to address these challenges to enhance the performance of the AI system.

## Getting Started

To run the project, follow these steps:

Clone the repository to your local machine.
Ensure you have the required dependencies installed. You can install them using pip:

pip install numpy pandas scikit-learn tensorflow keras matplotlib imbalanced-learn

Download the CCD dataset and place it in the project directory.
Run the credit_card_default_prediction.py script.

## Project Structure

The project directory contains the following files:

    Model.py: Python script containing the implementation of the deep learning model for credit card default prediction.
    CCD.xls: Dataset file containing credit card default data.
    README.md: This README file providing an overview of the project.
    Report.pd: The report and evaluation of the model

## Model Architecture

The deep learning model architecture consists of several layers including dense, batch normalization, dropout, and LSTM layers. The model is designed to effectively capture patterns in the input data and make accurate predictions.
Training and Evaluation

The model is trained using the Adam optimizer and binary cross-entropy loss function. Training progress and performance metrics such as accuracy, precision, recall, and F1-score are monitored and evaluated to assess the model's performance.
Results

The performance of the model is visualized using accuracy and loss plots. Additionally, key performance metrics including accuracy, loss score, precision, recall, and F1-score are reported to provide insights into the model's effectiveness in predicting credit card defaults.
