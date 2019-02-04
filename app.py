import os
import json
import numpy as np
import pandas as pd

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

# Import Flask & pymongo
from flask import Flask, jsonify, render_template, redirect, request
import pymongo

# Start Flask application:
app = Flask(__name__)

##################################################
# Use PostgreSQL for database:
##################################################

# Define environment variables

username = os.getenv("USERNAME_DB")
password = os.getenv("PASSWORD_DB")

# For now use csvs
path_to_athletes = os.path.join("data", "athletes_status.csv")
path_to_normal = os.path.join("data", "500_Person_Gender_Height_Weight_Index.csv")

##################################################
# Function to connect to the database
##################################################
"""
def dbConnect():

    ### For now, pull csvs

    return connection
"""
##################################################
# Function to concat data according to sport
##################################################

def pullData(sport):

	# Extract height, weight, sex, according to desired sport
	# Extract ordinary people and concat to one dataframe

    # For now, just pull data from csv
    athletes = pd.read_csv(path_to_athletes)
    ordinary = pd.read_csv(path_to_normal)

    # Use function input <sport> to breakdown athletes dataframe
    filter_ = "Sport"

    # athletes_sport dataframe adjustments
    athletes_sport = athletes.loc[athletes[filter_] == sport].copy()
    athletes_sport.drop(columns=["Unnamed: 0", "Age","Year","Season","Sport","Event"], inplace=True)

    # ordinary dataframe adjustments
    ordinary.rename(columns={"Gender":"Sex"}, inplace=True)

    # define function to rename sex field to M/F
    def gender_match(row):
        if row["Sex"] == "Male":
            val = "M"
        else:
            val = "F"
        return val

    ordinary["Sex"] = ordinary.apply(gender_match, axis=1)

    # Add status to all ordinary and remove index.
    ordinary["Status"] = "Ordinary"
    ordinary.drop(columns="Index", inplace=True)

    # concat dataframe of ordinary people and athletes
    people_df = pd.concat([athletes_sport, ordinary])

    return people_df

##################################################
# Function for Logistical Regression Model
##################################################

def LogRegression(height, weight, sex, sport):

	# consensus = Model.predict(height, weight, sex, sport)

    people_df = pullData(sport)

    X = people_df.drop(columns="Status")
    y = people_df.Status
    
    n = pd.get_dummies(X.Sex)
    X = pd.concat([X,n], axis=1)
    X.drop(["Sex"], inplace=True, axis=1)

    # Develop Logistical Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    classifier = LogisticRegression()

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    # Determine confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=["Ordinary", "Athlete"]).ravel()

    # Save model scoring to a dict
    modelDict = {
        "precision":np.round(precision_score(y_test, predictions, average="weighted"),4),
        "recall":np.round(recall_score(y_test, predictions, average="weighted"),4),
        "f1_score":np.round(f1_score(y_test, predictions, average="weighted"),4),
        "true_positive":tp,
        "true_negative":tn,
        "false_positive":fp,
        "false_negative":fn,
    }

    # # may need to adjust height and weight given metric used:
    # cm = np.round(height * 2.54, 0)
    # kg = np.round(weight * 0.453592, 0)

    # predict subject based on model
    if sex == "M":
        subject = np.array([[height, weight, 0, 1]]).astype(np.float64)
    elif sex == "F":
        subject = np.array([[height, weight, 1, 0]]).astype(np.float64)
    else:
        return "broken_ifstatement", "broken_ifstatement"

    try:
        consensus = classifier.predict(subject)[0]
    except:
        return "broken_here", "broken_here"

    return consensus, modelDict

##################################################
# Routes leading to templates:
##################################################

@app.route("/", methods=["GET", "POST"])
def index():
    """Return the homepage."""
    if request.method == "POST":
        userHeight = request.form["subjectHeight"]
        userWeight = request.form["subjectWeight"]
        userSex = request.form["subjectGender"]
        userSport = request.form["subjectSport"]

        # Function for Logistical Regression
        consensus, modelDict = LogRegression(userHeight, userWeight, userSex, userSport)

        return render_template("index.html", consensus=consensus, modelDict=modelDict)

    return render_template("index.html")

if __name__ == "__main__":
    app.run()