import pyrebase

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

# Initialize Firebase
firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

# Function to retrieve data from Firebase
def get_data_from_firebase(node):
    fb_input = db.child(node).get()
    if fb_input.val() is None:
        print("No data found at the specified node.")
        return None
    return fb_input.val()

# Function to update data in Firebase
def update_data_in_firebase(node, new_value):
    db.child(node).set(new_value)
    print(f"Data updated at node '{node}'.")

# Example usage
node_path = "/specialistSelected/userWants"  # Replace with your node path in Firebase
fb_input = get_data_from_firebase(node_path)

if fb_input:
    print("Data retrieved from Firebase:")
    print(fb_input)
    update_data_in_firebase(node_path, "")