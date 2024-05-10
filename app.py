from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('final_data.csv')
df = pd.DataFrame(df)

# Fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
ingredient_vectors = vectorizer.fit_transform(df['clean_ingredients'])

@app.route('/')
def introduction():
    return render_template('introduction.html')

@app.route('/recommendations', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']

        # Check if user input is empty or contains only commas
        if not user_input.replace(',', ' ').strip():
            return render_template('index.html', error="Please provide a valid list of ingredients.")

        user_vector = vectorizer.transform([user_input])

        cosine_similarities = cosine_similarity(user_vector, ingredient_vectors).flatten()

        N = 5
        top_indices = cosine_similarities.argsort()[-N:][::-1]
        top_recipes = df.iloc[top_indices]

        return render_template('index.html', user_input=user_input, top_recipes=top_recipes)
    return render_template('index.html')

from flask import jsonify

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback = request.form['feedback']
    # Process the feedback (e.g., store it in a database, use it to improve recommendations, etc.)
    feedback_message = "Feedback received. Thank you for your input!"
    return render_template('index.html', feedback_message=feedback_message)

if __name__ == '__main__':
    app.run(debug=True)
