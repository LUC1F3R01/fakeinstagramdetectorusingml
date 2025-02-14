import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        profile_pic = float(request.form['profile_pic'])
        username_len = float(request.form['username_len'])
        fullname_words = int(request.form['fullname_words'])
        fullname_len = float(request.form['fullname_len'])
        name_username = int(request.form['name_username'])
        description_len = int(request.form['description_len'])
        external_url = int(request.form['external_url'])
        private = int(request.form['private'])
        posts = int(request.form['posts'])
        followers = int(request.form['followers'])
        follows = int(request.form['follows'])

        # Prepare the input data (ensure it matches the expected number of features)
        input_data = np.array([[profile_pic, username_len, fullname_words, fullname_len, 
                                name_username, description_len, external_url, private, 
                                posts, followers, follows]])

        # Convert the input data to a pandas DataFrame with the feature names
        feature_names = model.feature_names_in_
        input_df = pd.DataFrame(input_data, columns=feature_names)

        # Make a prediction
        prediction = model.predict(input_df)
        
     
        if(prediction==0):
            result = 'Real'
        else:
            result = 'Fake'
        
        return render_template('index.html', prediction_text=f'The account is {result}')

if __name__ == "__main__":
    app.run(debug=True)
