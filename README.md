MACHINE LEARNING LAB FAT PREP
MultiVariable Regression
# Step 1: Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# Step 2: Define input variables (X1, X2)
X1 = np.array([0, 5, 15, 25, 35, 45, 55, 60])
X2 = np.array([1, 1, 2, 5, 11, 15, 34, 35])
# Target variable
y = np.array([4, 5, 20, 14, 32, 22, 38, 43])
# Step 3: Combine inputs into single matrix
X = np.column_stack((X1, X2))
# Step 4: Train model
model = LinearRegression()
model.fit(X, y)
# Step 5: Predict output
y_pred = model.predict(X)
# Step 6: Print results
print("Intercept (b0):", model.intercept_)
print("Coefficient for X1 (b1):", model.coef_[0])
print("Coefficient for X2 (b2):", model.coef_[1])
print("R2 Score:", r2_score(y, y_pred))
Logistic Regression
# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, log_loss
# Step 2: Load the dataset
df = pd.read_csv("ChurnData.csv")
# Step 3: Convert text data into numbers
# (Machine learning works only with numbers)
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
 df[col] = le.fit_transform(df[col])
# Step 4: Separate input (X) and output (y)
X = df.drop("churn", axis=1) # all columns except target
y = df["churn"] # target column
# Step 5: Split data into training and testing
# (train = learning, test = checking accuracy)
train_test_split(X, y, test_size=0.2, random_state=42)
# Step 6: Scale the data
# (helps model perform better)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Step 7: Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Step 8: Make predictions
y_pred = model.predict(X_test) # predicted values
y_prob = model.predict_proba(X_test) # probability values
# Step 9: Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nLog Loss:", log_loss(y_test, y_prob))
Hierarchial + K means Clustering
# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
# Step 2: Load dataset
df = pd.read_csv("tested (3).csv")
# Step 3: Select important features
X = df[['Age', 'Fare', 'Pclass']]
# Step 4: Handle missing values
# (replace missing values with mean)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
# Step 5: Scale data
# (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# -------- Hierarchical Clustering --------
# Step 6: Create dendrogram
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(linked)
plt.title("Hierarchical Clustering")
plt.show()
# -------- K-Means Clustering --------
# Step 7: Find best k using elbow method
wcss = []
for i in range(1, 6):
 kmeans = KMeans(n_clusters=i, random_state=0)
 kmeans.fit(X_scaled)
 wcss.append(kmeans.inertia_)
plt.plot(range(1, 6), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()
# Step 8: Apply K-Means (k = 3)
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled)
# Step 9: Show result
print(df[['Age', 'Fare', 'Pclass', 'Cluster']].head())
SVM
# Step 1: Import libraries
from sklearn import svm
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# =========================================
# PART 1: Linear Dataset (Basic SVM)
# =========================================
# Step 2: Create linear dataset
X, y = make_classification(n_samples=300, n_features=2, random_state=42)
# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Linear kernel
model_linear = svm.SVC(kernel='linear')
model_linear.fit(X_train, y_train)
# =========================================
# PART 2: Non-Linear SVM (RBF Kernel)
# =========================================
# Create non-linear dataset (circles)
X_rbf, y_rbf = make_circles(n_samples=300, noise=0.05)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_rbf, y_rbf, test_size=0.3)
model_rbf = svm.SVC(kernel='rbf')
model_rbf.fit(Xr_train, yr_train)
# =========================================
# PART 3: Non-Linear SVM (Polynomial Kernel)
# =========================================
# Create non-linear dataset (moons)
X_poly, y_poly = make_moons(n_samples=300, noise=0.1)
Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_poly, y_poly, test_size=0.3)
model_poly = svm.SVC(kernel='poly', degree=3)
model_poly.fit(Xp_train, yp_train)
# =========================================
# PART 4: Evaluation
# =========================================
# Linear accuracy
y_pred_linear = model_linear.predict(X_test)
print("Linear Accuracy:", accuracy_score(y_test, y_pred_linear))
# RBF accuracy
y_pred_rbf = model_rbf.predict(Xr_test)
print("RBF Accuracy:", accuracy_score(yr_test, y_pred_rbf))
# Polynomial accuracy
y_pred_poly = model_poly.predict(Xp_test)
print("Polynomial Accuracy:", accuracy_score(yp_test, y_pred_poly))
Boosting Models
# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
# Step 2: Load dataset
df = pd.read_csv("tested.csv")
# Step 3: Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median()) # fill missing age
df['Fare'] = df['Fare'].fillna(df['Fare'].median()) # fill missing fare
# Simplify Cabin column (take first letter)
df['Cabin'] = df['Cabin'].fillna('Missing').str[0]
# Step 4: Drop unwanted columns
df = df.drop(['Name', 'Ticket'], axis=1)
# Step 5: Convert categorical data to numbers
df = pd.get_dummies(df, drop_first=True)
# Step 6: Split input and output
X = df.drop(['PassengerId', 'Survived'], axis=1)
y = df['Survived']
# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# -------- Model 1: AdaBoost --------
ada = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=100)
ada.fit(X_train, y_train)
# -------- Model 2: XGBoost --------
xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
# -------- Model 3: Gradient Boost --------
gb = GradientBoostingClassifier(n_estimators=100)
gb.fit(X_train, y_train)
# Step 7: Evaluate models
print("AdaBoost:\n", classification_report(y_test, ada.predict(X_test)))
print("XGBoost:\n", classification_report(y_test, xgb.predict(X_test)))
print("Gradient Boost:\n", classification_report(y_test, gb.predict(X_test)))
# Step 8: Print accuracy
print("\nTrain Accuracy:")
print("Ada:", ada.score(X_train, y_train))
print("XGB:", xgb.score(X_train, y_train))
print("GB:", gb.score(X_train, y_train))
print("\nTest Accuracy:")
print("Ada:", ada.score(X_test, y_test))
print("XGB:", xgb.score(X_test, y_test))
print("GB:", gb.score(X_test, y_test))
# =========================================
# AdaBoost Parameter Tuning
# =========================================
# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Step 2: Load dataset
df = pd.read_csv("tested.csv")
# Step 3: Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
# Simplify Cabin column
df['Cabin'] = df['Cabin'].fillna('Missing').str[0]
# Step 4: Drop unwanted columns
df = df.drop(['Name', 'Ticket'], axis=1)
# Step 5: Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)
# Step 6: Define X and y
X = df.drop(['PassengerId', 'Survived'], axis=1)
y = df['Survived']
# Step 7: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Step 8: Try different parameters
depths = [1, 3, 5]
estimators = [2, 5, 10]
learning_rates = [0.001, 0.01]
# Step 9: Train models and print accuracy
for d in depths:
 for n in estimators:
 for lr in learning_rates:
 model = AdaBoostClassifier(
 estimator=DecisionTreeClassifier(max_depth=d),
 n_estimators=n,
 learning_rate=lr
 )
 model.fit(X_train, y_train)
 acc = accuracy_score(y_test, model.predict(X_test))
 print("Depth:", d, "| Estimators:", n, "| LR:", lr, "| Accuracy:", acc)
Neural Network
# Step 1: Import libraries
import pandas as pd
import numpy as np
# Step 2: Load dataset
df = pd.read_csv("fitness_dataset.csv")
# Step 3: Preprocessing
# Convert categorical to numbers
df['gender'] = df['gender'].map({'M': 1, 'F': 0}).fillna(0)
# Fill missing values
df = df.fillna(df.mean(numeric_only=True))
# Split input and output
X = df.drop('is_fit', axis=1).values
y = df['is_fit'].values.reshape(-1, 1)
# Normalize data
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
# Step 4: Define sigmoid function
def sigmoid(x):
 return 1 / (1 + np.exp(-x))
# Step 5: Initialize weights
input_size = X.shape[1]
weights = np.random.randn(input_size, 1)
# Step 6: Training (Forward + Backprop)
epochs = 50
lr = 0.01
for i in range(epochs):
 # Forward pass
 z = np.dot(X, weights)
 output = sigmoid(z)
 # Error calculation
 error = y - output
 # Backpropagation
 d_output = error * output * (1 - output)
 weights += np.dot(X.T, d_output) * lr
 # Print loss
 if (i+1) % 10 == 0:
 loss = np.mean(error**2)
 print("Epoch:", i+1, "Loss:", loss)
# Step 7: Prediction
pred = (output > 0.5).astype(int)
# Step 8: Accuracy
accuracy = np.mean(pred == y)
print("Accuracy:", accuracy)
