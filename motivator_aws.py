import boto3
import logging
import streamlit as st
import os
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()

# Access environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")


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
SYSTEM_TRAITS = """You are an advanced language model capable of detecting subtle emotional
nuances in text. Your goal is to identify and understand the emotions conveyed in user
statements, and provide relevant guidance from The Bible"""

SYSTEM_PROMPT = """You are a compassionate assistant that provides spiritual comfort and guidance."""

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
        region_name=os.getenv("AWS_REGION", "us-west-2"),
    )

    # Test connection
    # bedrock_client.list_foundation_models()
except Exception as e:
    logger.error(f"Failed to initialize Bedrock client: {str(e)}")
    st.error(
        "Failed to connect to AWS Bedrock. Please check your credentials and connection."
    )

MODEL_ID = "us.deepseek.r1-v1:0"

# Defining the chatbot's intents and responses
intents = {
    "excited": "For God did not give us a spirit of fear, but of power and of love and of a sound mind. - 2 Timothy 1:7",
    "satisfied": "And whatever you do, do it heartily, as for the Lord rather than for men. - Colossians 3:23",
    "joyful": "Rejoice in the Lord always. I will say it again: Rejoice! - Philippians 4:4",
    "proud": "For we are His workmanship, created in Christ Jesus for good works, which God prepared beforehand that we should walk in them. - Ephesians 2:10",
    "frustrated": "And we know that in all things God works for the good of those who love him, who have been called according to his purpose. - Romans 8:28",
    "anxious": "Cast your cares on the Lord and he will sustain you; he will never let the righteous be shaken. - Psalm 55:22",
    "overwhelmed": "Come to me, all you who are weary and burdened, and I will give you rest. - Matthew 11:28",
    "bored": "And whatever you do, do it heartily, as for the Lord rather than for men. - Colossians 3:23",
}

# Default response for unspecified emotions
default_response = "Even though I might not fully understand your current emotion, remember that God is always with you. \nHere's a verse for you: \nTrust in the Lord with all your heart and lean not on your own understanding. - Proverbs 3:5"

# system_traits = "You are an advanced language model capable of detecting subtle emotional nuances in text. Your goal is to identify and understand the emotions conveyed in user statements, and provide relevant guidance The Bible"


def generate_conversation(system_prompt, user_message):
    """
    Handles the conversation generation with Bedrock API
    """
    # messages = [
    #    {"role": "system", "content": [{"text": system_prompt}]},
    #    {"role": "user", "content": [{"text": user_message}]},
    # ]

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


# "You are an advanced language model capable of detecting subtle emotional nuances in text. Your goal is to identify and understand the emotions conveyed in user statements, and provide relevant guidance The Bible"
# "text": f"The user says: {user_input}. Based on this statement, determine the emotions involved and Provide Bible verse relevant to this emotion."


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
            user_message=f"The user says: {user_input}. Based on this statement, determine the emotions involved and Provide Bible verse relevant to this emotion.",
        )

        if not response or "output" not in response:
            return "error", "Received invalid response from model"

        # Extract response text
        ai_response = response["output"]["message"]["content"][0]["text"].lower()

        # Check for emotions in the AI response
        detected_emotions = [
            emotion for emotion in intents.keys() if emotion in ai_response
        ]
        if detected_emotions:
            # Return the first detected emotion and its corresponding verse
            return detected_emotions[0], intents[detected_emotions[0]]
        else:
            # Generate response for unmapped emotion
            emotion, response = generate_response_for_unmapped_emotion(ai_response)
            return (
                emotion if emotion else "unknown",
                response if response else default_response,
            )

    except Exception as e:
        logger.error(f"Error in get_chatbot_response: {e}")
        return "error", f"Error in processing: {e}"


def generate_response_for_unmapped_emotion(detected_emotion):
    try:
        response = generate_conversation(
            system_prompt=SYSTEM_PROMPT,
            user_message=f"The detected emotion is {detected_emotion}. Provide Bible verse relevant to this emotion.",
        )
        return detected_emotion, response["output"]["message"]["content"][0]["text"]

    except Exception as e:
        logger.error(f"Error in generate_response_for_unmapped_emotion: {e}")
        return None, f"Error in generating response: {e}"


# Streamlit application
st.title("Welcome to the Faith-based Motivator Chatbot!")
st.write(
    "Describe your current feelings, and receive encouraging Bible verse to get you going."
)

# Initialize session state for first submission
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False

# User input with a unique key
user_input = st.text_input("How are you feeling today?", key="user_input_key")

# Generate response when the user submits input
if st.button("Get Bible Verse"):
    if user_input:
        emotion, response = get_chatbot_response(user_input)
        if emotion:
            st.write(f"Detected emotion: **{emotion.capitalize()}**")
        st.write(f"Here's a Bible verse for you: {response}")
        st.session_state["submitted"] = True  # Mark as submitted
    else:
        st.write("Please enter your feeling to receive a Bible verse.")

# Show the option to share another feeling only after the first submission
if st.session_state["submitted"]:
    another = st.radio(
        "Would you like to share another feeling?",
        ("Yes", "No"),
        key="another_feeling_key",
    )
    if another == "Yes":
        st.write("I'm here to listen. Lay it on me.")
        st.session_state["submitted"] = False  # Reset for new input
    else:
        st.write("Thank you for chatting. May you have a blessed day!")
