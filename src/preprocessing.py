from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.

    # Separate columns according to the number of categories
    categorical_columns = working_train_df.select_dtypes('object').columns
    two_categories_columns = [col for col in categorical_columns if working_train_df[col].nunique() == 2]
    multiple_categories_columns = [col for col in categorical_columns if working_train_df[col].nunique() > 2]

    # 1. Instantiate encoders
    ordinal_encoder = OrdinalEncoder()
    one_hot_encoder = OneHotEncoder()

    # 2. Fit and transform Ordinal Encoder with columns with 2 Categories, 
    # however, I don't see an order in this columns to make them ordinal  
    working_train_df[two_categories_columns] = ordinal_encoder.fit_transform(working_train_df[two_categories_columns])
    working_val_df[two_categories_columns] = ordinal_encoder.transform(working_val_df[two_categories_columns])
    working_test_df[two_categories_columns] = ordinal_encoder.transform(working_test_df[two_categories_columns])

    # 3. Fit and Transform the columns with multiple categories
    onehot_encoded_train = one_hot_encoder.fit_transform(working_train_df[multiple_categories_columns]).toarray()
    onehot_encoded_val = one_hot_encoder.transform(working_val_df[multiple_categories_columns]).toarray()
    onehot_encoded_test = one_hot_encoder.transform(working_test_df[multiple_categories_columns]).toarray()

    # 4. Drop the columns with multiple categories and concatenate the encoded columns
    working_train_df_encoded = working_train_df.drop(columns=multiple_categories_columns)
    working_val_df_encoded = working_val_df.drop(columns=multiple_categories_columns)
    working_test_df_encoded = working_test_df.drop(columns=multiple_categories_columns)


    onehot_encoded_train_df = pd.DataFrame(onehot_encoded_train,
                                        columns=one_hot_encoder.get_feature_names_out(multiple_categories_columns))
    onehot_encoded_val_df = pd.DataFrame(onehot_encoded_val,
                                    columns=one_hot_encoder.get_feature_names_out(multiple_categories_columns))
    onehot_encoded_test_df = pd.DataFrame(onehot_encoded_test,
                                    columns=one_hot_encoder.get_feature_names_out(multiple_categories_columns))        

    # Concat the encoded columns
    working_train_df_encoded = pd.concat([working_train_df_encoded.reset_index(drop=True), onehot_encoded_train_df.reset_index(drop=True)], axis=1, ignore_index=True)
    working_val_df_encoded = pd.concat([working_val_df_encoded.reset_index(drop=True), onehot_encoded_val_df.reset_index(drop=True)], axis=1, ignore_index=True)
    working_test_df_encoded = pd.concat([working_test_df_encoded.reset_index(drop=True), onehot_encoded_test_df.reset_index(drop=True)], axis=1)


    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.

    simple_imputer = SimpleImputer(strategy='median')
    working_train_df_imputed = pd.DataFrame(simple_imputer.fit_transform(working_train_df_encoded))
    working_val_df_imputed = pd.DataFrame(simple_imputer.transform(working_val_df_encoded))
    working_test_df_imputed = pd.DataFrame(simple_imputer.transform(working_test_df_encoded))

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.
    
    min_max_scaler = MinMaxScaler()
    working_train_df = pd.DataFrame(min_max_scaler.fit_transform(working_train_df_imputed), columns=working_train_df_imputed.columns)
    working_val_df = pd.DataFrame(min_max_scaler.transform(working_val_df_imputed), columns=working_val_df_imputed.columns)
    working_test_df = pd.DataFrame(min_max_scaler.transform(working_test_df_imputed), columns=working_test_df_imputed.columns)
    
    working_train = working_train_df.to_numpy()
    working_val = working_val_df.to_numpy()
    working_test = working_test_df.to_numpy()

    return working_train, working_val, working_test

