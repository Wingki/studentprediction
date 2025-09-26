import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # Drop ID if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    # Categorical and numerical columns
    cat_cols = [
        'gender', 'province', 'home_language', 'parent_edu', 'income_bracket',
        'study_mode', 'faculty', 'course_difficulty', 'instructor_exp', 'internet_access'
    ]
    num_cols = [c for c in df.columns if c not in cat_cols + ['pass']]
    # Impute numeric
    imputer_num = SimpleImputer(strategy='mean')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])
    # Impute categorical
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    # One-hot encode categoricals
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_cat = encoder.fit_transform(df[cat_cols])
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    X_cat = pd.DataFrame(X_cat, columns=cat_feature_names)
    # Normalize numericals
    scaler = StandardScaler()
    X_num = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)
    # Combine
    X = pd.concat([X_num, X_cat], axis=1)
    y = df['pass'].astype(int)
    return X, y

def load_and_preprocess(path):
    df = pd.read_csv(path)
    return preprocess_data(df)