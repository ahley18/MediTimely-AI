import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
import pyrebase
import time
import uuid

# Firebase configuration
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
try:
    data = pd.read_csv(r"files/hospital_doctors5.csv", encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv(r"files/hospital_doctors5.csv", encoding='latin1')
data['doctor_specialization'] = data['doctor_specialization'].str.lower()

# Define features and target variable
X = data[['hospital_distance(km)', 'num_of_available_beds', 'experience(years)', 'rating(1-5)', 'num_testimonies']]
y = data['hospital_name']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=21)

# Train the Logistic Regression model with hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(estimator=LogisticRegression(random_state=21, max_iter=1000), param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_weighted')
grid_search.fit(X_train, y_train)
best_lr = grid_search.best_estimator_

# Predict on the test set
y_pred = best_lr.predict(X_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred, average='weighted')
print('Logistic Regression Model')
print(f"Accuracy: {accuracy}")
print(f"F1-score: {f1}")

# Function to retrieve data from Firebase and save as a variable
def get_data_from_firebase(node):
    try:
        fb_input = db.child(node).get()
        if fb_input.val() is None:
            print("No data found at the specified node.")
            return None
        return fb_input.val()
    except Exception as e:
        print(f"Error retrieving data from Firebase: {e}")
        return None

# Function to clear node in Firebase
def update_data_in_firebase(node, new_value):
    try:
        db.child(node).set(new_value)
        print(f"Data updated at node '{node}'.")
    except Exception as e:
        print(f"Error updating data in Firebase: {e}")

# Function to convert all numpy int64 to Python int in a dictionary
def convert_int64_to_int(d):
    for k, v in d.items():
        if isinstance(v, pd.Series):
            d[k] = v.item()  # If value is a single item pd.Series, extract the item
        elif isinstance(v, (np.int64, np.float64)):
            d[k] = int(v)  # Convert numpy int64 or float64 to Python int
        elif isinstance(v, dict):
            convert_int64_to_int(v)  # Recursively handle nested dictionaries
    return d

# Function to predict the nearest hospitals for given features
def predict_nearest_hospitals(distance, available_beds, specialization, top_n=20):
    # Filter the data by the required specialization
    filtered_data = data[data['doctor_specialization'] == specialization]

    if filtered_data.empty:
        return {}

    # Sort by hospital distance first, then by available beds
    filtered_data = filtered_data.sort_values(by=['hospital_distance(km)', 'num_of_available_beds'])

    # Get the top N nearest hospitals
    top_hospitals = filtered_data.head(top_n)

    # Group by hospital name and collect doctor details
    grouped_hospitals = top_hospitals.groupby('hospital_name')

    results = []
    for hospital_name, group in grouped_hospitals:
        hospital_info = {
            "hospitalName": hospital_name,
            "location": "Manila, Philippines",  # This is a placeholder; update as needed
            "distance": group['hospital_distance(km)'].iloc[0],  # Set distance from the top hospital
            "doctors": [],
            "clinic_id": group['hospital_id'].iloc[0]  # Get the clinic_id from the group
        }
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            doctor_info = {
                "rank": rank,  # Correctly assign the rank starting from 1
                "name": row['doctor_name'],
                "specialty": row['doctor_specialization'],
                "experience": f"{row['experience(years)']} years",
                "rating": row['rating(1-5)'],
                "testimonies": row['num_testimonies'],
                "image": f"require('../assets/images/doctor{rank}.jpg')"  # Example image path
            }
            hospital_info["doctors"].append(doctor_info)

        results.append(hospital_info)

    # Sort results by distance
    results_sorted = sorted(results, key=lambda x: x['distance'])

    return results_sorted

# Loop to continuously ask for doctor specialization and update Firebase
while True:
    node_path = "/specialistSelected/userWants"  # Replace with your node path in Firebase
    fb_input = get_data_from_firebase(node_path)

    if fb_input:
        update_data_in_firebase(node_path, "")  # Clear the node in Firebase
        specialization = fb_input.lower()

        if specialization == "exit":
            print("Exiting loop.")
            break

        distance = 10  # example distance in km
        available_beds = 5  # example number of available beds
        top_hospitals = predict_nearest_hospitals(distance, available_beds, specialization)

        if not top_hospitals:
            print("Couldn't find doctor with specialization:", specialization)
        else:
            # Upload sorted data to Firebase
            for hospital_data in top_hospitals:
                # Convert int64 to int in hospital_data
                hospital_data = convert_int64_to_int(hospital_data)

                # Generate a unique ID
                unique_id = uuid.uuid4().hex

                # Push data under the unique ID
                db.child(f"clinic_test/{unique_id}").set(hospital_data)

                # Print hospital and doctor details
                print(f"Hospital: {hospital_data['hospitalName']}, Distance: {hospital_data['distance']} km")
                for doctor in hospital_data['doctors']:
                    print(f"Rank: {doctor['rank']}, Doctor: {doctor['name']}, Specialty: {doctor['specialty']}, "
                          f"Experience: {doctor['experience']}, Rating: {doctor['rating']}, "
                          f"Testimonies: {doctor['testimonies']}")

    else:
        print("No valid data retrieved from Firebase.")

    # Sleep for a short period to avoid rapid polling
    time.sleep(10)

