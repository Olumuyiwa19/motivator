import boto3
import logging
import streamlit as st
import os
import json
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import re

# Loading environment variables
load_dotenv()

# Access environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
MODEL_ID = os.getenv("MODEL_ID")
#"amazon.nova-micro-v1:0",   "us.deepseek.r1-v1:0"

# Validate environment variables
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION]):
    raise ValueError(
        "Missing required environment variables. Please check your .env file."
    )

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Define system traits as a constant
SYSTEM_TRAITS = """You are an advanced language model capable of detecting subtle emotional nuances in text.
Your goal is to accurately identify and name the exact emotion(s) conveyed in user statements
(e.g., fear, anger, sadness, joy, hope, guilt, etc.), and provide relevant guidance from
The Bible that addresses or speaks to those emotions with wisdom, encouragement, and truth.
Return emotion detected, Bible, Message in a machine readable json format only like the example below
{"<<emotion detected>>":{"Bible":["<<bible verse - bible text>>"],"Message": "<<concise Message of encouragement based on the emotion detected and the bible verses>>"}}
"""

try:
    # Create Bedrock Runtime client with error handling
    custom_config = Config(
        connect_timeout=840,
        read_timeout=840,
        retries={"max_attempts": 3},  # Add retry mechanism
    )

    client = boto3.client("bedrock-runtime")

    # Test connection
    # bedrock_client.list_foundation_models()
except Exception as e:
    logger.error(f"Failed to initialize Bedrock client: {str(e)}")
    st.error(
        "Failed to connect to AWS Bedrock. Please check your credentials and connection."
    )

# Defining the chatbot's intents and responses
intents = {
    "excited":{
        "bible":["2 Timothy 1:7 - For God did not give us a spirit of fear, but of power and of love and of a sound mind"],
        "message": "Remember heaven rejoice with you, and the earth be glad"
        },
    "satisfied":{
        "bible":["Colossians 3:23 - And whatever you do, do it heartily, as for the Lord rather than for men."],
        "message": "And whatever you do, do it heartily, as for the Lord rather"
    }
}

# Default response for unspecified emotions
#default_response = "Even though I might not fully understand your current emotion, remember that God is always with you. \nHere's a verse for you: \nTrust in the Lord with all your heart and lean not on your own understanding. - Proverbs 3:5"

def generate_conversation(system_prompt, user_message):
    """
    Handles the conversation generation with Bedrock API
    """
    # Combine system prompt and user message
    combined_message = f"{system_prompt}\n\nUser: {user_message}"

    messages = [{"role": "user", "content": [{"text": combined_message}]}]

    inf_params = {"temperature": 0.3, "maxTokens": 300, "topP": 0.1}
    additionalModelRequestFields = {
    "inferenceConfig": {
         "topK": 20
    }
}
    try:
        response = client.converse(
            modelId=MODEL_ID,
            messages=messages,
            inferenceConfig=inf_params,
            additionalModelRequestFields=additionalModelRequestFields
        )

        # Log token usage if available
        if "usage" in response:
            token_usage = response.get("usage", {})
            for key in ["inputTokens", "outputTokens", "totalTokens"]:
                logger.info(f"{key}: {token_usage.get(key, 'N/A')}")

        return response
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        logger.error(f"Bedrock API error: {error_code} - {error_message}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

def get_chatbot_response(user_input):
    """
    Process user input and generate appropriate response
    """
    if not user_input:
        return "error", "Please provide some input"

    try:
        # Send request to the model through Bedrock
        response = generate_conversation(
            system_prompt=SYSTEM_TRAITS,
            user_message=f"The user says: {user_input}",
        )

        if not response or "output" not in response:
            return "error", "Received invalid response from model"

        # Extract response text
        ai_response = response["output"]["message"]["content"][0]["text"]


        # Try multiple patterns to extract JSON content
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
            r'(\{(?:"[^"]*"\s*:\s*\{[^}]*\}(?:,|))*\})',  # Direct JSON object
            r'(\{.*\})'  # Any JSON-like structure (fallback)
        ]

        parsed_json = None
        for pattern in json_patterns:
            match = re.search(pattern, ai_response, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1)
                    parsed_json = json.loads(json_str)
                    break
                except json.JSONDecodeError:
                    continue

        if parsed_json:
            ai_response = parsed_json
        else:
            # If no JSON found, try to create a simple JSON structure from the text
            try:
                # Look for emotion keywords in the response
                emotion_keywords = ["excited", "satisfied", "joyful", "proud",
                                   "frustrated", "anxious", "overwhelmed", "bored",
                                   "happy", "sad", "angry", "fearful", "hopeful"]

                found_emotion = None
                for emotion in emotion_keywords:
                    if emotion.lower() in ai_response.lower():
                        found_emotion = emotion
                        break

                if not found_emotion:
                    found_emotion = "unspecified"

                # Extract Bible verse if present (looking for common Bible verse patterns)
                bible_match = re.search(r'([1-3]?\s*\w+\s+\d+:\d+(?:-\d+)?)\s*[-–]\s*([^"]*)', ai_response)
                bible_verse = bible_match.group(0) if bible_match else "Proverbs 3:5 - Trust in the Lord with all your heart"

                # Create a simple JSON structure
                ai_response = {
                    found_emotion: {
                        "Bible": [bible_verse],
                        "Message": ai_response[:100]  # Use first 100 chars as message
                    }
                }
                logger.info(f"Created fallback JSON structure with emotion: {found_emotion}")
            except Exception as e:
                logger.warning(f"Failed to parse response: {e}")
                return "error", "Invalid response format", "Please try again"

        ai_emotions = ', '.join(ai_response.keys())

        # Check if the AI response contains any of the predefined intents
        # If it does, extract the first one
        captured_intent_input = [
            emotion for emotion in intents if emotion in ai_emotions
        ]
        if captured_intent_input:
            bibleVerse = intents[captured_intent_input[0]]['Bible'][0]
            message = intents[captured_intent_input[0]]['Message']

            # Return the first detected emotion and its corresponding verse
            return captured_intent_input[0], bibleVerse, message
        else:
            # If no emotion is detected, return a default response
            first_emotion = next(iter(ai_response))
            # Fix case sensitivity in key names
            bible_key = next((k for k in ai_response[first_emotion].keys() if k.lower() == "bible"), "Bible")
            message_key = next((k for k in ai_response[first_emotion].keys() if k.lower() == "message"), "Message")

            bibleVerse = ai_response[first_emotion][bible_key][0] #if isinstance(ai_response[first_emotion][bible_key], list) else ai_response[first_emotion][bible_key]
            message = ai_response[first_emotion][message_key]
            return first_emotion, bibleVerse, message

    except Exception as e:
        logger.error(f"Error in get_chatbot_response: {e}")
        return "error", f"Error in processing: {e}"

# Streamlit application configuration
st.title("Welcome to the Faith-based Motivator Chatbot!")
st.subheader("Your Personal Encouragement Assistant")
# Disclaimer
st.write(
    "This chatbot is a personal project designed to help you find encouragement and guidance through the Bible. It is not meant to replace professional help, but rather to provide you with a source of inspiration and support."
)
st.write(
    "Describe your current feelings, and receive encouraging Bible verse to get you going."
)

# Initialize session state for first submission
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False
# Initialize session state for resources
if "resources_key" not in st.session_state:
    st.session_state["resources_key"] = False

# User input with a unique key
user_input = st.text_input("How are you feeling today?", key="user_input_key")

# Generate response when the user submits input
if st.button("Get Bible Verse"):
    if user_input:
        emotion, bible, message = get_chatbot_response(user_input)
        if emotion != "error":
            # Display the detected emotion and Bible verse
            st.write(f"I can persive that you are feeling: **{emotion.capitalize()}**")
            st.write(f"Here's a Bible verse for you: {bible}")
            st.write(f"Message: {message}")
            st.session_state["submitted"] = True  # Mark as submitted
        else:
            st.error(bible)
            if message:
                st.info(message)

    else:
        st.warning("Please enter your feeling to receive a Bible verse.")

# Show the option to share feedback after submission
if st.session_state["submitted"]:
    feedback = st.radio(
        "Do you feel encouraged by this response?",
        options=("Select an option", "Yes", "No"),
        key="feedback_key",
        index=0, # Default index to first option"
    )

    if feedback == "Yes":
        st.success("Thank you for your feedback! We are glad to hear that you feel encouraged.")
        st.session_state["resources_key"] = True

    elif feedback == "No":
        resources_response = st.radio(
            "Would you like to see other helpful resources?",
            options=("Select an option", "Yes", "No"),
            key="resources_radio",
            index=0, # Default index to first option"
        )

        if resources_response == "Yes":
            # Only show resources when user explicitly asks for them
            show_resources = st.button("View Resources")
            if show_resources:
                st.markdown("""
                    ### Helpful Resources
                    - [Bible Gateway](https://www.biblegateway.com/)
                    - [YouVersion Bible App](https://www.youversion.com/the-bible-app/)
                    - [Faith-based Counseling Services](https://www.psychologytoday.com/us/therapists/religious-spirituality)
                    - [Christian Meditation and Mindfulness](https://www.christianmeditation.com/)
                """)

                if st.button("Start Over"):
                    st.session_state["submitted"] = False
                    st.session_state["resources_key"] = False
                    st.rerun()

        elif resources_response == "No":
            st.write("Thank you for chatting. May you have a blessed day!")
            if st.button("Start New Conversation"):
                st.session_state["submitted"] = False
