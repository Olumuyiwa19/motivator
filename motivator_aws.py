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
#"amazon.nova-micro-v1:0"

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

    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        config=custom_config,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )

    # Test connection
    # bedrock_client.list_foundation_models()
except Exception as e:
    logger.error(f"Failed to initialize Bedrock client: {str(e)}")
    st.error(
        "Failed to connect to AWS Bedrock. Please check your credentials and connection."
    )

# Defining the chatbot's intents and responses
intent = {
    "excited": "For God did not give us a spirit of fear, but of power and of love and of a sound mind. - 2 Timothy 1:7",
    "satisfied": "And whatever you do, do it heartily, as for the Lord rather than for men. - Colossians 3:23",
    "joyful": "Rejoice in the Lord always. I will say it again: Rejoice! - Philippians 4:4",
    "proud": "For we are His workmanship, created in Christ Jesus for good works, which God prepared beforehand that we should walk in them. - Ephesians 2:10",
    "frustrated": "And we know that in all things God works for the good of those who love him, who have been called according to his purpose. - Romans 8:28",
    "anxious": "Cast your cares on the Lord and he will sustain you; he will never let the righteous be shaken. - Psalm 55:22",
    "overwhelmed": "Come to me, all you who are weary and burdened, and I will give you rest. - Matthew 11:28",
    "bored": "And whatever you do, do it heartily, as for the Lord rather than for men. - Colossians 3:23",
}

intents = {
    "excited":{
    "bible":["2 Timothy 1:7 - For God did not give us a spirit of fear, but of power and of love and of a sound mind"],
    "message": "Remember heaven rejoice with you, and the earth be glad"},
    "satisfied":{
        "bible":["Colossians 3:23 - And whatever you do, do it heartily, as for the Lord rather than for men."],
        "message": "And whatever you do, do it heartily, as for the Lord rather"
    }
    }

# Default response for unspecified emotions
default_response = "Even though I might not fully understand your current emotion, remember that God is always with you. \nHere's a verse for you: \nTrust in the Lord with all your heart and lean not on your own understanding. - Proverbs 3:5"

# system_traits = "You are an advanced language model capable of detecting subtle emotional nuances in text. Your goal is to identify and understand the emotions conveyed in user statements, and provide relevant guidance The Bible"


def generate_conversation(system_prompt, user_message):
    """
    Handles the conversation generation with Bedrock API
    """
    # Combine system prompt and user message
    combined_message = f"{system_prompt}\n\nUser: {user_message}"

    messages = [{"role": "user", "content": [{"text": combined_message}]}]

    inference_config = {"temperature": 0.5, "maxTokens": 512, "topP": 0.9}

    try:
        response = bedrock_client.converse(
            modelId=MODEL_ID, messages=messages, inferenceConfig=inference_config
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
        # Send request to DeepSeek through Bedrock
        response = generate_conversation(
            system_prompt=SYSTEM_TRAITS,
            user_message=f"The user says: {user_input}",
        )


        if not response or "output" not in response:
            return "error", "Received invalid response from model"

        # Extract response text
        ai_response = response["output"]["message"]["content"][0]["text"]


        # Use regex to extract the JSON content
        match = re.search(r'```json\s*(\{.*?\})\s*```', ai_response, re.DOTALL)
        if match:
            json_str = match.group(1)
            ai_response = json.loads(json_str)
        else:
            logger.warning("No valid JSON found in response.")

            return "error", "Invalid response format", "Please try again"

        ai_emotions = ', '.join(ai_response.keys())

        # Check if the AI response contains any of the predefined intents
        # If it does, extract the first one
        captured_intent_input = [
            emotion for emotion in intents if emotion in ai_emotions
        ]
        if captured_intent_input:

            bibleVerse=intents[captured_intent_input[0]]['Bible'][0]
            message = intents[captured_intent_input[0]]['Message']

            # Return the first detected emotion and its corresponding verse
            return captured_intent_input[0], bibleVerse, message
        else:
            # If no emotion is detected, return a default response
            #return None, default_response
            first_emotion = next(iter(ai_response))
            bibleVerse=ai_response[first_emotion]['Bible'][0]
            message = ai_response[first_emotion]['Message']
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

        #st.session_state["submitted"] = True  # Mark as submitted

    else:
        st.warning("Please enter your feeling to receive a Bible verse.")

# Show the option to share another feeling only after the first submission
if st.session_state["submitted"]:
    feedback = st.radio(
        "Do you feel encouraged by this response?",
        ("Yes", "No"),
        key="feedback_key",
    )
    if feedback == "No":
        resources_response = st.radio(
            "Would you like to see other helpful resources?",
            ("Yes", "No"),
            key="resources_radio"
        )

        if resources_response == "Yes":
            st.markdown("""
                ### Helpful Resources
                - [Bible Gateway](https://www.biblegateway.com/)
                - [YouVersion Bible App](https://www.youversion.com/the-bible-app/)
                - [Faith-based Counseling Services](https://www.psychologytoday.com/us/therapists/religious-spirituality)
            """)

            if st.button("Start Over"):
                st.session_state["submitted"] = False
        else:
            st.write("Thank you for chatting. May you have a blessed day!")
            if st.button("Start New Conversation"):
                st.session_state["submitted"] = False
