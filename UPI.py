import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('upii.csv')

# Dropping non-numeric columns that won't be useful for PCA
columns_to_drop = ['TransactionID', 'UserID', 'SenderUPIID', 'ReceiverUPIID', 'DeviceID', 'IPAddress', 'GeoLocation', 'TransactionType', 'BankAccountNumber']
data_numeric = data.drop(columns=columns_to_drop)

# Encoding categorical columns: 'Status'
label_encoder = LabelEncoder()
data_numeric['Status'] = label_encoder.fit_transform(data_numeric['Status'])

# Separating features and target variable
X = data_numeric.drop('IsFraud', axis=1)
y = data_numeric['IsFraud']

# Identify and encode all remaining non-numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = label_encoder.fit_transform(X[col])

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# Creating a DataFrame with the PCA results
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['IsFraud'] = y.values

# Plotting the PCA results
plt.figure(figsize=(10, 6))
plt.scatter(pca_df[pca_df['IsFraud'] == 0]['PC1'], pca_df[pca_df['IsFraud'] == 0]['PC2'], label='Non-Fraud', alpha=0.5)
plt.scatter(pca_df[pca_df['IsFraud'] == 1]['PC1'], pca_df[pca_df['IsFraud'] == 1]['PC2'], label='Fraud', alpha=0.5, color='r')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Transaction Data')
plt.legend()
plt.grid(True)
plt.show()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying PCA
pca = PCA(n_components=2)  # Reduce to 2 components
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Training the Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_pca, y_train)

# Making predictions
y_pred = classifier.predict(X_test_pca)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
