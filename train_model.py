import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import pickle

def create_train_test_data(dataset):

    # Select 80% of the data for training while maintaining the original index for separation.
    data_train = dataset.sample(frac=0.8, random_state=30)
    
    # Select the lines that are NOT in the training using the original index.
    data_test = dataset.drop(data_train.index)

    # We reset the index to save clean files.
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    # Save the files
    data_train.to_csv('train.csv', index=False)
    data_test.to_csv('test.csv', index=False)

    print(f"Train data for modeling: {data_train.shape}")
    print(f"Test data for predictions: {data_test.shape}")

def train_model(x_train, y_train):
    print("Training the model ...")
    model = Pipeline(steps=[
        ("label encoding", OneHotEncoder(handle_unknown='ignore')),
        ("tree model", LinearRegression())
    ])
    model.fit(x_train, y_train)
    return model

def accuracy(model, x_test, y_test):
    print("Testing the model ...")
    predictions = model.predict(x_test)
    tree_mse = mean_squared_error(y_test, predictions)
    tree_rmse = np.sqrt(tree_mse)
    return tree_rmse

def export_model(model):
    pkl_path = 'model.pkl'
    with open(pkl_path, 'wb') as file:
        pickle.dump(model, file)
        print(f"Model saved at {pkl_path}")

def main():
    # Load the clean data. I removed index_col=0 to avoid conflicts with the previously saved CSV file.
    data = pd.read_csv('cleaned_data.csv', keep_default_na=False)

    # Divide training and testing
    create_train_test_data(data)

    # load data for trainning
    train = pd.read_csv('train.csv', keep_default_na=False)
    x_train = train.drop(columns=['SalePrice'])
    y_train = train['SalePrice']

    # load data for testing
    test = pd.read_csv('test.csv', keep_default_na=False)
    x_test = test.drop(columns=['SalePrice'])
    y_test = test['SalePrice']

    # Train and Test
    model = train_model(x_train, y_train)
    rmse_test = accuracy(model, x_test, y_test)

    print(f"Average Price Test: {y_test.mean()}")
    print(f"RMSE: {rmse_test}")

    # save the model
    export_model(model)

if __name__ == '__main__':
    main()