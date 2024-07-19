import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import pyrebase

firebaseConfig = {
    'apiKey': "AIzaSyBmZM9DZ9tDP8tdPCs9qaJ0SF7mukjuMcg",
    'authDomain': "meditimely.firebaseapp.com",
    'projectId': "meditimely",
    'storageBucket': "meditimely.appspot.com",
    'messagingSenderId': "20774129356",
    'appId': "1:20774129356:web:5c12d17dd4890417effc69",
    'databaseURL': "https://meditimely-default-rtdb.firebaseio.com/"
}

firebase = pyrebase.initialize_app(firebaseConfig)

db = firebase.database()

# Load the dataset
data = pd.read_csv("files/sample_hospital_data_100.csv")
data['doctor_specialization'] = data['doctor_specialization'].str.lower()

# Define features and target variable
X = data[['hospital_distance(km)', 'num_of_available_beds']]
y = data['hospital_name']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=21)

# Train the logistic regression model
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy}")
print(f"F1-score: {f1}")


# Function to predict the nearest hospitals for given features
def predict_nearest_hospitals(distance, available_beds, specialization, top_n=10):
    # Filter the data by the required specialization
    filtered_data = data[data['doctor_specialization'] == specialization]

    if filtered_data.empty:
        return []

    # Sort by hospital distance first, then by available beds
    filtered_data = filtered_data.sort_values(by=['hospital_distance(km)', 'num_of_available_beds'])

    # Get the top N nearest hospitals
    top_hospitals = filtered_data.head(top_n)

    results = []
    for rank, (_, row) in enumerate(top_hospitals.iterrows(), 1):
        hospital_name = row['hospital_name']
        doctor_name = row['doctor_name']
        results.append({
            "rank": rank,
            "hospital_name": hospital_name,
            "doctor_name": doctor_name,
            "distance": row['hospital_distance(km)'],
            "available_beds": row['num_of_available_beds']
        })

    return results


# Example usage
distance = 10  # example distance in km
available_beds = 5  # example number of available beds
specialization = input('Input doctor specialization: ').lower()  # example specialization
top_hospitals = predict_nearest_hospitals(distance, available_beds, specialization)

# Upload data to Firebase under a single node with rankings
for i, hospital in enumerate(top_hospitals, 1):
    print(f"{i}. Hospital: {hospital['hospital_name']}, Doctor: {hospital['doctor_name']}, "
          f"Distance: {hospital['distance']} km, Available Beds: {hospital['available_beds']}")

    dataDB = {
        'rank': hospital['rank'],
        'hospital': hospital['hospital_name'],
        'doctor': hospital['doctor_name'],
        'distance': hospital['distance'],
        'bed_count': hospital['available_beds']
    }
    db.push(dataDB)
