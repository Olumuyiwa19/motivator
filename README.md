# Faith-based Motivator Chatbot

A Streamlit-based chatbot that provides Biblical encouragement and verses based on your emotional state. The chatbot uses AWS Bedrock with the DeepSeek model to detect emotions and deliver relevant scripture verses.

## Features

- Emotion detection from user input
- Biblical verses curated for various emotional states
- Support for multiple emotional states including:
  - Excited
  - Satisfied
  - Joyful
  - Proud
  - Frustrated
  - Anxious
  - Overwhelmed
  - Bored
- Fallback responses for unrecognized emotions
- Built with AWS Bedrock and DeepSeek's language model
- Interactive chat interface using Streamlit

## Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- AWS credentials configured

## Installation

1. Clone the repository:
```sh
git clone <repository-url>
cd motivator-chatbot
```

2. Install required dependencies:
```sh
pip install -r requirements.txt
```

3. Create a `.env` file based on the example:
```sh
cp example.env .env
```

4. Configure your AWS credentials in the `.env` file:
```
AWS_ACCESS_KEY_ID="your_access_key_here"
AWS_SECRET_ACCESS_KEY="your_secret_access_key_here"
AWS_REGION=your_preferred_region_here
```

## Usage

1. Start the Streamlit application:
```sh
streamlit run motivator_aws.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Enter how you're feeling in the text input field

4. Click "Get Bible Verse" to receive an encouraging Biblical response

## Project Structure

- `motivator_aws.py` - Main application file with AWS Bedrock integration
- `requirements.txt` - Python dependencies
- `.env` - Environment variables configuration
- `example.env` - Template for environment variables

## Dependencies

- boto3==1.34.69
- streamlit==1.32.2
- python-dotenv==1.0.1
- botocore==1.34.69

## Environment Variables

| Variable | Description |
|----------|-------------|
| AWS_ACCESS_KEY_ID | Your AWS access key |
| AWS_SECRET_ACCESS_KEY | Your AWS secret access key |
| AWS_REGION | AWS region (e.g., us-west-2) |

## Error Handling

The application includes comprehensive error handling for:
- Missing environment variables
- AWS connection issues
- Invalid responses from the model
- Input validation
