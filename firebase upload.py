import pyrebase

usr_name = input('Name: ')
usr_age = input('Age: ')
usr_addr = input('Address: ')

firebaseConfig = {
    'apiKey': "AIzaSyBmZM9DZ9tDP8tdPCs9qaJ0SF7mukjuMcg",
    'authDomain': "meditimely.firebaseapp.com",
    'projectId': "meditimely",
    'storageBucket': "meditimely.appspot.com",
    'messagingSenderId': "20774129356",
    'appId': "1:20774129356:web:5c12d17dd4890417effc69",
    'databaseURL': "https://meditimely-default-rtdb.firebaseio.com/"  # Ensure this line is added
}

firebase = pyrebase.initialize_app(firebaseConfig)

db = firebase.database()

# Define a reference to a specific node where you want to push the data

# Push data
data = {'name': usr_name, 'age': usr_age, 'address': usr_addr}
db.push(data)
