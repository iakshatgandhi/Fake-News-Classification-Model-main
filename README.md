# Fake News Classification Project

==========================

**Project Overview**

This project utilizes natural language processing (NLP) and machine learning algorithms to classify news articles as real or fake. It features a user-friendly web interface built with Streamlit for effortless interaction.

**Features**

*   **Data Preprocessing:** Thorough cleaning and preprocessing of data to prepare it for modeling.
*   **Text Vectorization:** Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features.
*   **Classification Models:** Explores multiple models including Naive Bayes, Logistic Regression, and Random Forest for optimal classification performance.
*   **Model Evaluation:** Compares and analyzes the performance of different models using metrics like accuracy, precision, recall, and F1-score.
*   **Visualizations:** Generates informative visualizations to understand data distribution, model performance, and feature importance (for Random Forest).
*   **Web Interface:** Streamlit-based interface allows for real-time classification of news articles directly through a web browser.

**Installation**

1.  **Clone this repository.**
2.  **Install required packages:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download NLTK data:**

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

**Usage**

1.  **Prepare dataset:** Place files named `True.csv` and `Fake.csv` containing your real and fake news data in the project directory.
2.  **Run data preprocessing and model training script:**

    ```bash
    python data_cleaning.py
    ```

3.  **Launch Streamlit app:**

    ```bash
    streamlit run app.py
    ```

4.  **Open web browser:** Navigate to the provided URL (usually `http://localhost:8501`) to access the web interface.

**File Descriptions**

*   `data_cleaning.py`: This script handles data preprocessing, model training, and evaluation.
*   `app.py`: This script builds the Streamlit web application for user interaction.
*   `fake_news_model.pkl`: Stores the best-performing trained model for real-time classification.
*   `tfidf_vectorizer.pkl`: Stores the fitted TF-IDF vectorizer used for text feature extraction.

**Model Performance**

The project evaluates the performance of multiple classification models:

*   Naive Bayes
*   Logistic Regression
*   Random Forest

It analyzes the models using various metrics:

*   Accuracy
*   Precision
*   Recall
*   F1-score

**Visualizations**

The project generates insightful visualizations to understand the data and model behavior:

*   Model performance comparison
*   Dataset distribution (real vs. fake news)
*   Text length distribution
*   Learning curves
*   Feature importance (for Random Forest)

These visualizations are accessible through the Streamlit web interface.

**Web Interface**

The Streamlit web interface allows users to:

*   Input news articles for real-time classification.
*   View the classification result (real or fake) along with a confidence score.
*   Explore model performance metrics and visualizations.

**Future Improvements**

*   Implement cross-validation for robust model evaluation.
*   Experiment with deep learning models like LSTM and BERT.
*   Add features like sentiment analysis and named entity recognition.
*   Enhance the web interface with interactive elements.
*   Implement user feedback collection for continuous improvement.

**Contributing**

We welcome contributions! Feel free to submit a Pull Request to enhance this project.

**License**

This project is open-source under the MIT License.
