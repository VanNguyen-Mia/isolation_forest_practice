import joblib
from src.module_1 import load_data, find_anomaly, transform_data, train_and_save_model, classify_new_applicant

def main():

    # Define dataset_path and model_save_path
    dataset_path = './data/loan_data.csv'
    model_save_path = './saved_model/rf_model.joblib'
    train_columns_path = './saved_model/trained_columns.joblib'

    # Step 1: Load data
    df = load_data(dataset_path)

    # Step 2: Find anomaly
    df = find_anomaly(df)

    # Step 3: Transform data
    df = transform_data(df)

    # Step 4: Train and save model
    accuracy = train_and_save_model(df, model_save_path, train_columns_path)
    print(f"Model trained and saved with accuracy: {accuracy:.3f}")

if __name__ == '__main__':
    main()