Movie Review Sentiment Analysis
A beginner-friendly data science project that trains a machine learning model to classify movie reviews as either positive or negative. This project uses Python and the Scikit-learn library to process text data and build a predictive model.

🚀 Features
Data Preprocessing: Cleans and prepares raw text data for machine learning.

Model Training: Builds a Logistic Regression model on the IMDb dataset.

Sentiment Prediction: Classifies new, unseen movie reviews.

Interactive Mode: Allows you to type in your own reviews and get instant sentiment predictions.

📊 Dataset
This project uses the IMDb Dataset of 50K Movie Reviews from Kaggle. The dataset contains 50,000 movie reviews, each labeled with a positive or negative sentiment.

The script can be configured to download this dataset automatically using the Kaggle API.

🛠️ Installation
Follow these steps to set up the project environment.

Clone the repository (or download the files):

git clone <your-repository-url>
cd movie-review-sentiment-analysis

Create a Python virtual environment:

python -m venv .venv

Activate the virtual environment:

On Windows:

.\.venv\Scripts\Activate.ps1

On macOS/Linux:

source .venv/bin/activate

Install the required libraries:

pip install pandas matplotlib seaborn scikit-learn nltk

▶️ How to Run
Once the setup is complete, you can run the main script from your terminal:

python main.py

The script will first download the necessary NLTK data packages (stopwords and wordnet), then it will train the model and display the evaluation results (including accuracy and a confusion matrix). Finally, it will enter an interactive mode where you can input your own reviews.

📄 License
This project is licensed under the MIT License. See the LICENSE.md file for details.
