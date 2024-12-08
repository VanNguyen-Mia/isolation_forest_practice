# Data Processing
import pandas as pd

# Modelling
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib


def load_data(path:str) -> pd.DataFrame:
    """
    Load the dataset from a path

    Args:
        path (str): path to the dataset

    Returns:
        pd.DataFrame: output dataframe of the dataset
    """
    df = pd.read_csv(path)
    return df

def find_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find anomaly and return a clean data

    Args:
        df (pd.DataFrame): original data

    Returns:
        pd.DataFrame: clean data
    """
    model = IsolationForest(n_estimators=100, max_samples=256, contamination=0.0005, random_state=42)
    model.fit(df[['person_age']])
    df['scores']=model.decision_function(df[['person_age']])
    df['anomaly']=model.predict(df[['person_age']])
    df[['person_age', 'scores', 'anomaly']].head()
    anomaly=df[['person_age', 'scores','anomaly']].loc[df['anomaly']==-1]
    print(anomaly)
    df = df.loc[df['anomaly']!=-1]
    df = df.drop(columns = ['scores', 'anomaly'])
    return df

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform object-type data to number-type data

    Args:
        df (pd.DataFrame): data with original type

    Returns:
        pd.DataFrame: transformed data ready for random forest
    """
    # person_gender column
    df['person_gender'] = df['person_gender'].map({'male': 0, 'female': 1})
    # previous_loan_defaults_on_file column
    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})
    # person_education column
    dummy_person_education = pd.get_dummies(df['person_education'], dtype=int)
    dummy_person_education.rename(columns={'Associate':'edu_associate',	
                                       'Bachelor':'edu_bachelor', 
                                       'Doctorate':'edu_doctorate', 
                                       'High School':'edu_highschool',	
                                       'Master':'edu_master'}, inplace=True)
    # person_home_ownership column
    dummy_home_ownership = pd.get_dummies(df['person_home_ownership'], dtype = int)
    dummy_home_ownership.rename(columns={'MORTGAGE':'ownership_mortgage',	
                                     'OTHER':'ownership_other',	
                                     'OWN':'ownership_own',	
                                     'RENT': 'ownership_rent'}, inplace=True)
    # loan_intent column
    dummy_loan_intent = pd.get_dummies(df['loan_intent'], dtype=int)
    dummy_loan_intent.rename(columns={'DEBTCONSOLIDATION':'intent_debtconsolidation',
                                  	'EDUCATION':'intent_education',
                                    'HOMEIMPROVEMENT':'intent_homeimprovement',
                                    'MEDICAL':'intent_medical',
                                    'PERSONAL':'intent_personal',
                                    'VENTURE':'intent_venture'}, inplace=True)
    df = pd.concat([df, dummy_home_ownership, dummy_loan_intent,
                dummy_person_education], axis=1)
    df = df.drop(columns={'person_education', 
                      'person_home_ownership', 'loan_intent',})
    return df

def train_and_save_model(df: pd.DataFrame, model_path: str, train_columns_path: str) -> float:
    """
    Train Random Forest and save the model for future prediction
    Args:
        df (pd.DataFrame): cleaned and transformed dataset
        model_path (str): path to save the trained model
        train_columns_path (str): path to the train columns' names

    Returns:
        pd.DataFrame: 
    """
    features = df.drop('loan_status', axis=1)
    target = df['loan_status']

    # split the data into training and test sets
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)
    
    # fitting and evaluating the Random Forest Model
    rf_model = RandomForestClassifier()
    rf_model.fit(features_train, target_train)

    target_pred = rf_model.predict(features_test)

    accuracy = accuracy_score(target_test, target_pred)

    # save the trained model
    joblib.dump(rf_model, model_path)
    print(f"Model saved to {model_path}")

    # save column names
    trained_columns = list(features.columns)
    joblib.dump(trained_columns, train_columns_path)
    print(f"Columns names saved to {train_columns_path}")

    return accuracy

def classify_new_applicant(applicant_data: dict, model_path: str, train_columns_path: str) -> str:
    """
    Classify a new loan applicant using the trained model

    Args:
        applicant_data (dict): dictionary containing applicant's information
        model_path (str): path to the saved model
        train_columns_path (str): path to the train columns names
        transform_data (_type_): function transforming the data

    Returns:
        str: classification result: approved or rejected
    """
    applicant_df = pd.DataFrame([applicant_data])
    applicant_df = transform_data(applicant_df)

    rf_model = joblib.load(model_path)
    trained_columns = joblib.load(train_columns_path)  # Trained column names saved during training

    # Align the transformed data with the trained model's expected columns
    for col in trained_columns:
        if col not in applicant_df.columns:
            applicant_df[col] = 0  # Add missing columns with default value 0

    # Reorder the columns to match the trained model
    applicant_df = applicant_df[trained_columns]

    prediction = rf_model.predict(applicant_df)[0]
    if prediction == 1:
        return "Approved"
    else:
        return "Rejected"

                                    