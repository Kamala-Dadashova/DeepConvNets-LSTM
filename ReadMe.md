
## Projects

### 1. CNN Model Modification and Ensemble Learning

**Problem Statement:** 
I will use mnist data to perform the followings, 
- Use "same" padding for all convolution layers and add batch normalization after each layer.
- Add two convolution layers with 32 filters each (3x3, stride 1), each followed by a max-pooling layer (2x2).
- Use ReLU activations for all convolution layers.
- Change the number of epochs to 50 and set the validation split to 30%.
- Use a callback to save the model with the best validation accuracy and evaluate this model's testing accuracy.
- Use only the first 1,000 images in the training set.
- Compare the out-of-sample test accuracy for a single CNN with an ensemble of 5, 10, or 20 CNNs using bootstrap aggregation and majority voting.

### 2. Sentiment Classification Using IMDB Dataset

**Problem Statement:** 
- Load the IMDB dataset for sentiment classification with `num_words=5000`.
- Pad each sequence to a maximum length of 500.
- Create an LSTM model with the following configuration:
  - Embedding layer mapping words to a vector of length 16.
  - One LSTM layer with 64 hidden states.
  - One dense layer with a sigmoid output for binary sentiment classification (good/bad).
- Compile the model with binary cross-entropy loss and Adam optimizer.
- Train for 10 epochs and report the out-of-sample test set accuracy.
- Repeat the process with different embedding vector lengths (8, 16, 32) and hidden state numbers (16, 32, 64, 128). Report the results in a table.

### 3. Decision Tree and Random Forest Classification

**Problem Statement:**
- Use the Pima Indian Diabetes dataset to train decision trees.
- Perform 5-fold cross-validation to tune hyperparameters and report the cross-validation error on the test set.
- Train decision trees using `sklearn.tree.DecisionTreeClassifier` and tune `max_depth` and `min_samples_split`.
- Train a random forest model using `sklearn.ensemble.RandomForestClassifier` with `n_estimators` set to 10, 50, and 100, using the optimal hyperparameters from the decision tree tuning.

## Getting Started

To get started with these projects, you need to have Python installed along with the necessary libraries. You can install the required libraries using pip:

```bash
pip install pandas numpy plotly tensorflow keras scipy scikit-learn jupyter

## Libraries and Tools Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive graphing
- **tensorflow**: Machine learning framework
- **keras**: Deep learning API
- **scipy**: Scientific computing
- **scikit-learn (sklearn)**: Machine learning library
- **Python**: Programming language
- **Jupyter Notebook**: Interactive computing environment
