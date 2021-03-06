import os
import json
import numpy as np
import pandas as pd

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

# Import Flask 
from flask import Flask, jsonify, render_template, redirect, request

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

def pullData(sport, event):

	# Extract height, weight, sex, according to desired sport
	# Extract ordinary people and concat to one dataframe

    # For now, just pull data from csv
    athletes = pd.read_csv(path_to_athletes)
    ordinary = pd.read_csv(path_to_normal)

    # Use function input <sport> to breakdown athletes dataframe
    athletes_sport = athletes.loc[athletes["Sport"] == sport].copy()

    # Use <event> to break down df even further.
    if event != "":
        athletes_sport = athletes_sport.loc[athletes_sport["Event"] == event]


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

def LogRegression(height, weight, sex, sport, event):

	# consensus = Model.predict(height, weight, sex, sport)

    people_df = pullData(sport, event)

    X = people_df.drop(columns="Status")
    y = people_df.Status
    
    n = pd.get_dummies(X.Sex)
    X = pd.concat([X,n], axis=1)
    X.drop(["Sex"], inplace=True, axis=1)

    # Develop Logistical Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    # Logistical Regression:
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    # # Random Forest
    # classifier = RandomForestClassifier(n_estimators=200)
    # classifier.fit(X_train, y_train)
    # predictions = classifier.predict(X_test)

    # # Support Vector Machines
    # classifier = SVC(kernel='linear')
    # classifier.fit(X_train, y_train)
    # predictions = classifier.predict(X_test)

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

    # adjust height to represent centimeters.
    height_in = height.split("'")
    height_in = int(height_in[0])*12 + int(height_in[1])

    # may need to adjust height and weight given metric used:
    cm = np.round(height_in * 2.54, 0)
    kg = np.round(int(weight) * 0.453592, 0)

    # predict subject based on model
    if sex == "M":
        subject = np.array([[cm, kg, 0, 1]]).astype(np.float64)
    elif sex == "F":
        subject = np.array([[cm, kg, 1, 0]]).astype(np.float64)
    else:
        return "broken_ifstatement", "broken_ifstatement"

    consensus = classifier.predict(subject)[0]

    if consensus == "Athlete":
    	consensus = f"Congrats! You could compete in {sport}"
    else:
    	consensus = f"Your body type does not match athletes in {sport}"

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
        userEvent = request.form["subjectEvent"]

        # Function for Logistical Regression
        consensus, modelDict = LogRegression(userHeight, userWeight, userSex, userSport, userEvent)

        return render_template("index.html", consensus=consensus, modelDict=modelDict)
        
    # define default model
    modelDict = {
        "precision":"",
        "recall":"",
        "f1_score":"",
        "true_positive":"",
        "true_negative":"",
        "false_positive":"",
        "false_negative":"",
    }

    return render_template("index.html", modelDict=modelDict)

@app.route("/about/")
def about():
	""" Return the about page to explain data clensing and visuals """
	return render_template("about.html")


if __name__ == "__main__":
    app.run()