# Question-Generator-Application

## Overview
This application uses Generative AI to generate new programming questions based on provided samples.

## Requirements
- Python 3.x
- Flask
- pandas
- openai

## Setup
1. Install required packages:
    ```sh
    pip install flask pandas openai
    ```

2. Set up your OpenAI API key in the `app.py` file:
    ```python
    openai.api_key = 'your_openai_api_key_here'
    ```

3. Place your CSV file in the application directory.

4. Run the application:
    ```sh
    python app.py
    ```

## Usage
- Access the application at `http://127.0.0.1:5000`.
- Use the form to generate questions.
- Export questions in CSV or JSON format.

## API Endpoints
- `POST /generate`: Generates new questions based on the provided parameters.
- `POST /export`: Exports the questions in the specified format.
