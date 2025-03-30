import os
import boto3
import streamlit as st
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Loading environment variables from .env file
load_dotenv()

# Setting environment variables
#AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
#AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="us-west-2")

# Set the model ID, e.g., DeepSeek-R1 Model.
model_id = "us.deepseek.r1-v1:0"

# Creating Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-01"
)

# Defining the chatbot's intents and responses
intents = {
    "excited": "For God did not give us a spirit of fear, but of power and of love and of a sound mind. - 2 Timothy 1:7",
    "satisfied": "And whatever you do, do it heartily, as for the Lord rather than for men. - Colossians 3:23",
    "joyful": "Rejoice in the Lord always. I will say it again: Rejoice! - Philippians 4:4",
    "proud": "For we are His workmanship, created in Christ Jesus for good works, which God prepared beforehand that we should walk in them. - Ephesians 2:10",
    "frustrated": "And we know that in all things God works for the good of those who love him, who have been called according to his purpose. - Romans 8:28",
    "anxious": "Cast your cares on the Lord and he will sustain you; he will never let the righteous be shaken. - Psalm 55:22",
    "overwhelmed": "Come to me, all you who are weary and burdened, and I will give you rest. - Matthew 11:28",
    "bored": "And whatever you do, do it heartily, as for the Lord rather than for men. - Colossians 3:23"
}

# Default response for unspecified emotions
default_response = "Even though I might not fully understand your current emotion, remember that God is always with you. \nHere's a verse for you: \nTrust in the Lord with all your heart and lean not on your own understanding. - Proverbs 3:5"

def get_chatbot_response(user_input):
    # Creating messages for the API
    messages = [
        {"role": "system", "content": "You are an advanced language model capable of detecting subtle emotional nuances in text. Your goal is to identify and understand the emotions conveyed in user statements, and provide relevant guidance The Bible"},
        {"role": "user", "content": f"The user says: {user_input}. Based on this statement, determine the emotions involved and Provide Bible verse relevant to this emotion."}
    ]

    try:
        # Creating a chatbot response using Azure OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        # Extracting the emotion from the AI response
        ai_response = response.choices[0].message.content.lower()

        # Check for emotions in the AI response
        detected_emotions = [emotion for emotion in intents.keys() if emotion in ai_response]
        if detected_emotions:
            # For simplicity, we'll respond with the first detected emotion
            return detected_emotions[0], intents[detected_emotions[0]]

        # If a valid emotion is detected but not in intents, generate a response using the model's knowledge
        if any(emotion in ai_response for emotion in ["sad", "angry", "confused", "etc."]):
            return generate_response_for_unmapped_emotion(ai_response)

        # If no specific emotion is detected, return a default response
        return None, default_response

    except Exception as e:
        return None, f"Error in API call: {e}"

def generate_response_for_unmapped_emotion(detected_emotion):
    # Use a new prompt to generate a response for the unmapped emotion
    messages = [
        {"role": "system", "content": "You are a compassionate assistant that provides spiritual comfort and guidance."},
        {"role": "user", "content": f"The detected emotion is {detected_emotion}. Provide Bible verse relevant to this emotion."}
    ]

    try:
        # Generate a response using Azure OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        # Return the generated message
        return detected_emotion, response.choices[0].message.content

    except Exception as e:
        return None, f"Error in generating response: {e}"

# Streamlit application
st.title("Welcome to the Faith-based Motivator Chatbot!")
st.write("Describe your current feelings, and receive encouraging Bible verse to get you going.")

# Initialize session state for first submission
if 'submitted' not in st.session_state:
    st.session_state['submitted'] = False

# User input with a unique key
user_input = st.text_input("How are you feeling today?", key="user_input_key")

# Generate response when the user submits input
if st.button("Get Bible Verse"):
    if user_input:
        emotion, response = get_chatbot_response(user_input)
        if emotion:
            st.write(f"Detected emotion: **{emotion.capitalize()}**")
        st.write(f"Here's a Bible verse for you: {response}")
        st.session_state['submitted'] = True  # Mark as submitted
    else:
        st.write("Please enter your feeling to receive a Bible verse.")

# Show the option to share another feeling only after the first submission
if st.session_state['submitted']:
    another = st.radio("Would you like to share another feeling?", ('Yes', 'No'), key="another_feeling_key")
    if another == 'Yes':
        st.write("I'm here to listen. Lay it on me.")
        st.session_state['submitted'] = False  # Reset for new input
    else:
        st.write("Thank you for chatting. May you have a blessed day!")