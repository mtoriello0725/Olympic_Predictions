import os
import json
import numpy as np
import pandas as pd

# Import Flask & pymongo
from flask import Flask, jsonify, render_template, redirect, request
import pymongo

# Start Flask application:
app = Flask(__name__)

##################################################
# Use PostgreSQL for database:
##################################################

# Define environment variables

username = os.get_env("USERNAME_DB")
password = os.get_env("PASSWORD_DB")

def dbConnect(username, password)


##################################################
# Function for Logistical Regression Model
##################################################

def pullData(sport):

	# Extract height, weight, sex, according to desired sport
	# Extract ordinary people and concat to one dataframe

	# people_df

	return people_df


def LogRegression(height, weight, sex, sport):

	# consensus = Model.predict(height, weight, sex, sport)

	return consensus

##################################################
# Routes leading to templates:
##################################################

@app.route("/", methods=["GET", "POST"])
def index():
    """Return the homepage."""
    if request.method = "POST":
    	userHeight = request.form["userHeight"]
    	userWeight = request.form["userWeight"]
    	userSex = request.form["userSex"]
    	userSport = request.form["userSport"]

    	# Function for pulling data according to sport

    	# Function for Logistical Regression

    	return render_template("index.html", consensus=consensus)

    return render_template("index.html")

if __name__ == "__main__":
    app.run()