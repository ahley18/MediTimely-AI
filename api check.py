import re

# List of doctor titles to search for
doctor_titles = [
    "Dermatologist", "Radiologist", "Oncologist", "Neurologist",
    "General Surgeon", "Cardiologist", "Psychiatrist", "Orthopedist",
    "Endocrinologist", "Pediatrician"
]

# Join the titles into a single regex pattern
pattern = r'|'.join(doctor_titles)

# Paragraph to search in
text = "A dermatologist is a doctor who specializes in the diagnosis and treatment of skin diseases. They can diagnose and treat a variety of skin conditions, including rashes, acne, eczema, psoriasis, and skin cancer. If you have a rash, it is important to see a dermatologist to get a diagnosis and treatment. A dermatologist can help you determine the cause of your rash and recommend the best treatment."

# Search for the first occurrence of any doctor title
match = re.search(pattern, text, re.IGNORECASE)

if match:
    print(match.group())
else:
    print("No doctor title found.")
