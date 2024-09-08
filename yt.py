from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load dataset from Excel file
def load_data(file_path):
    return pd.read_excel(file_path)

# Train the model
def train_model(data):
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    model.fit(data['text'], data['category'])
    return model

# Function to predict the category of user input
def predict_category(model, user_input):
    return model.predict([user_input])[0]

# Custom responses for procurement and spending activities
def get_procurement_response(category):
    if category == 'spending_info':
        return "Your current spending limit is $5,000. Would you like to check recent expenses?"
    elif category == 'purchase_order_status':
        return "Your last purchase order (PO #1234) is currently being processed. Expected delivery is in 3 days."
    elif category == 'budget_check':
        return "You have $1,200 remaining in your Q3 budget. Be mindful of upcoming expenses."
    elif category == 'procurement_process':
        return "To initiate a procurement request, please fill out the procurement form and submit it to the approval committee."
    else:
        return "I'm not sure how to respond to that."

# Streamlit app
def main():
    st.title("Procurement and Spending Chatbot")
    st.write("Ask me anything related to your spending or procurement activities!")

    # Load the dataset
    file_path = '/mnt/data/chatbot_dataset.xlsx'  # Path to the uploaded Excel file
    data = load_data(file_path)

    # Train the model
    model = train_model(data)

    # User input
    user_input = st.text_input("You:", "")

    if st.button("Send"):
        if user_input.strip() != "":
            # Predict the category of the message
            category = predict_category(model, user_input)

            # Get the appropriate response
            bot_response = get_procurement_response(category)

            # Display the bot's response
            st.text_area("Bot:", bot_response, height=150)
        else:
            st.warning("Please enter a message.")

if __name__ == '__main__':
    main()
