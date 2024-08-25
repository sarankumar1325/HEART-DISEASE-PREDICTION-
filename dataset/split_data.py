import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = pd.read_csv("../data/heart.csv")
    train_data, test_data = train_test_split(data, test_size=0.33, random_state=42)
    train_data.to_csv("./heart_train.csv")
    test_data.to_csv("./heart_test.csv")