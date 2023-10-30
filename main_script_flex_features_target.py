import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError

# Suppress scikit-learn's warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Reading data from the Excel workbook
xls = pd.ExcelFile('data.xlsx')
df1 = pd.read_excel(xls, sheet_name='historical')

# Display column names and let user select features and target
print("Available columns:", df1.columns.tolist())
features = input("Please enter the feature columns separated by commas: ").split(',')
target = input("Please enter the target column: ")

# Define the prediction_header here
prediction_header = f"f:{'-'.join(features)};t:{target}"

# Drop columns not selected
columns_to_keep = features + [target]
df1 = df1[columns_to_keep]

# Splitting dataset 1 into features and target
X = df1.drop(columns=[target])
y = df1[target]

# Splitting dataset 1 into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Determine the most frequent category for each feature to set as reference
reference_categories = {feature: df1[feature].value_counts().idxmax() for feature in features}

# Creating a column transformer that applies one-hot encoding to categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(categories=[
            sorted(df1[feature].unique(), key=lambda x: (x != reference_categories[feature], x)) for feature in features
        ], drop='first'), features)
    ],
    remainder='passthrough'
)

# Creating a pipeline that first applies the column transformer and then applies logistic regression
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Training the model
pipeline.fit(X_train, y_train)

# Getting feature names after one-hot encoding
feature_names = (pipeline.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .get_feature_names_out(features))

# Getting coefficients from the logistic regression model
coefficients = pipeline.named_steps['classifier'].coef_[0]

# Pairing feature names with their coefficients
coef_pairs = list(zip(feature_names, coefficients))

# Adding reference categories with a coefficient of 0
coef_pairs.extend([(feature + '_' + reference_categories[feature], 0) for feature in features])

# Sorting and printing coefficients by group
print("\nCoefficients:")
for feature in features:
    relevant_coef_pairs = [pair for pair in coef_pairs if feature in pair[0]]
    sorted_relevant_coef_pairs = sorted(relevant_coef_pairs, key=lambda x: x[1], reverse=True)
    for name, coef in sorted_relevant_coef_pairs:
        print(f"{name}: {coef:.4f}")

# Predicting on the validation set
y_val_pred = pipeline.predict(X_val)

# Calculating and printing the accuracy
accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {accuracy:.4f}")

# Predicting probabilities for dataset 2
df2 = pd.read_excel(xls, sheet_name='to_predict')
try:
    predicted_probabilities = pipeline.predict_proba(df2)[:, 1]
except Exception as e:
    if "Found unknown categories" in str(e):
        print("Warning: There are categories in the 'to_predict' dataset that were not present in the training data. These categories will be ignored during prediction.")
        encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
        encoder.handle_unknown = 'ignore'
        predicted_probabilities = pipeline.predict_proba(df2)[:, 1]
        encoder.handle_unknown = 'error'  # Resetting it back to 'error' for future safety

# Adding the probabilities to dataset 2
df2['predicted_conversion_probability'] = predicted_probabilities

# Load the existing 'to_predict' sheet
df_existing = pd.read_excel('data.xlsx', sheet_name='to_predict')



# Add the new prediction column to the existing DataFrame
df_existing[prediction_header] = predicted_probabilities

# Save the updated DataFrame back to the 'to_predict' sheet
with pd.ExcelWriter('data.xlsx', engine='openpyxl', mode='a') as writer:
    # First, we need to remove the existing 'to_predict' sheet to avoid duplication
    writer.book.remove(writer.book['to_predict'])
    df_existing.to_excel(writer, sheet_name='to_predict', index=False)

