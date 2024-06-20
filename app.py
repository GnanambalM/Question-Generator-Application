import os
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
import openai
import logging

app = Flask(__name__)

csv_file_path = "D://Iamneo//Parsed_Questions (1).csv"
questions_df = pd.read_csv(csv_file_path)

openai.api_key = 'sk-proj-hejDE3KLrHyHxeW2mYmiT3BlbkFJJBFAwpTPxacFcaZhQmPe'

logging.basicConfig(level=logging.DEBUG)

def generate_questions(topic, num_questions, difficulty, language):
    prompt = f"Generate {num_questions} {difficulty} questions on {topic} in {language}."
    logging.debug(f"Prompt for OpenAI: {prompt}")
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        logging.debug(f"OpenAI Response: {response}")

        questions = response['choices'][0]['message']['content'].strip().split('\n')
        logging.debug(f"Generated questions: {questions}")
        
        return questions
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI Error: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        logging.debug(f"Received data: {data}")
        topic = data.get('topic')
        num_questions = int(data.get('num_questions'))
        difficulty = data.get('difficulty')
        language = data.get('language')
        
        questions = generate_questions(topic, num_questions, difficulty, language)
        
        if questions:
            return jsonify({'questions': questions})
        else:
            return jsonify({'error': 'Could not generate questions'}), 500
    except Exception as e:
        logging.error(f"Error in /generate endpoint: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/export', methods=['POST'])
def export():
    try:
        data = request.json
        export_format = data.get('format', 'csv')
        
        if export_format == 'csv':
            export_path = 'exported_questions.csv'
            questions_df.to_csv(export_path, index=False)
        elif export_format == 'json':
            export_path = 'exported_questions.json'
            questions_df.to_json(export_path, orient='records')
        
        return send_file(export_path, as_attachment=True)
    except Exception as e:
        logging.error(f"Error in /export endpoint: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)

"""from flask import Flask, request, jsonify
import openai
import pandas as pd

app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = 'sk-proj-1S7rLZqc75o9Ke6XChrjT3BlbkFJDSll9JeZb3yKYmui5dC3'

# Function to generate questions
def generate_question(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7
    )
    question = response.choices[0].text.strip()
    return question

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    topic = data.get('topic')
    num_questions = data.get('num_questions', 1)
    difficulty = data.get('difficulty', 'Medium')
    language = data.get('language', 'Python')
    
    questions = []
    for _ in range(num_questions):
        prompt = f"Generate a {difficulty} programming question about {topic} in {language}."
        question = generate_question(prompt)
        questions.append({
            'topic': topic,
            'question': question,
            'difficulty': difficulty,
            'language': language
        })

    return jsonify({'questions': questions})

@app.route('/export', methods=['POST'])
def export():
    questions = request.json.get('questions', [])
    df = pd.DataFrame(questions)
    csv_file = 'generated_questions.csv'
    df.to_csv(csv_file, index=False)
    return jsonify({'message': f'Questions exported to {csv_file}'})

if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, request, jsonify
import openai
import pandas as pd
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Set your OpenAI API key
openai.api_key = 'sk-proj-1S7rLZqc75o9Ke6XChrjT3BlbkFJDSll9JeZb3yKYmui5dC3'

@app.route('/')
def home():
    return "Welcome to the Question Generator API. Use /generate to generate questions and /export to export questions."

# Function to generate questions
def generate_question(prompt):
    logging.debug(f"Generating question with prompt: {prompt}")
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        question = response.choices[0].text.strip()
        return question
    except Exception as e:
        logging.error(f"Error generating question: {e}")
        return "Error generating question"

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        topic = data.get('topic')
        num_questions = data.get('num_questions', 1)
        difficulty = data.get('difficulty', 'Medium')
        language = data.get('language', 'Python')

        questions = []
        for _ in range(num_questions):
            prompt = f"Generate a {difficulty} programming question about {topic} in {language}."
            question = generate_question(prompt)
            questions.append({
                'topic': topic,
                'question': question,
                'difficulty': difficulty,
                'language': language
            })

        return jsonify({'questions': questions})
    except Exception as e:
        logging.error(f"Error in /generate endpoint: {e}")
        return jsonify({'error': 'An error occurred while generating questions'}), 500

@app.route('/export', methods=['POST'])
def export():
    try:
        questions = request.json.get('questions', [])
        df = pd.DataFrame(questions)
        csv_file = "D://Iamneo//Parsed_Questions (1).csv"
        df.to_csv(csv_file, index=False)
        return jsonify({'message': f'Questions exported to {csv_file}'})
    except Exception as e:
        logging.error(f"Error in /export endpoint: {e}")
        return jsonify({'error': 'An error occurred while exporting questions'}), 500

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Error starting the Flask app: {e}")


from flask import Flask, request, jsonify, render_template_string
import openai
import pandas as pd
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Set your OpenAI API key
openai.api_key = 'your_openai_api_key'

@app.route('/')
def home():
    return '''
        <h1>Welcome to the Question Generator API</h1>
        <p>Use the forms below to test the endpoints.</p>
        <form action="/generate" method="post">
            <h2>Generate Questions</h2>
            <label for="topic">Topic:</label>
            <input type="text" id="topic" name="topic"><br><br>
            <label for="num_questions">Number of Questions:</label>
            <input type="number" id="num_questions" name="num_questions" value="1"><br><br>
            <label for="difficulty">Difficulty:</label>
            <input type="text" id="difficulty" name="difficulty" value="Medium"><br><br>
            <label for="language">Language:</label>
            <input type="text" id="language" name="language" value="Python"><br><br>
            <input type="submit" value="Generate">
        </form>
        <br>
        <form action="/export" method="post">
            <h2>Export Questions</h2>
            <label for="questions">Questions (JSON format):</label><br>
            <textarea id="questions" name="questions" rows="10" cols="50"></textarea><br><br>
            <input type="submit" value="Export">
        </form>
    '''

# Function to generate questions
def generate_question(prompt):
    logging.debug(f"Generating question with prompt: {prompt}")
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        question = response.choices[0].text.strip()
        return question
    except Exception as e:
        logging.error(f"Error generating question: {e}")
        return "Error generating question"

@app.route('/generate', methods=['POST'])
def generate():
    try:
        topic = request.form.get('topic')
        num_questions = int(request.form.get('num_questions', 1))
        difficulty = request.form.get('difficulty', 'Medium')
        language = request.form.get('language', 'Python')

        questions = []
        for _ in range(num_questions):
            prompt = f"Generate a {difficulty} programming question about {topic} in {language}."
            question = generate_question(prompt)
            questions.append({
                'topic': topic,
                'question': question,
                'difficulty': difficulty,
                'language': language
            })

        return jsonify({'questions': questions})
    except Exception as e:
        logging.error(f"Error in /generate endpoint: {e}")
        return jsonify({'error': 'An error occurred while generating questions'}), 500

@app.route('/export', methods=['POST'])
def export():
    try:
        questions_json = request.form.get('questions', '[]')
        questions = pd.read_json(questions_json)
        csv_file = "D://Iamneo//Parsed_Questions (1).csv"
        questions.to_csv(csv_file, index=False)
        return jsonify({'message': f'Questions exported to {csv_file}'})
    except Exception as e:
        logging.error(f"Error in /export endpoint: {e}")
        return jsonify({'error': 'An error occurred while exporting questions'}), 500

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Error starting the Flask app: {e}")


from flask import Flask, request, jsonify
import pandas as pd
import random
import openai

app = Flask(__name__)

# Load the CSV file into a DataFrame when the application starts
csv_file_path = "D://Iamneo//Parsed_Questions (1).csv"  # Update with your actual CSV file path
questions_df = pd.read_csv(csv_file_path)

# Set your OpenAI API key
openai.api_key = 'sk-proj-1S7rLZqc75o9Ke6XChrjT3BlbkFJDSll9JeZb3yKYmui5dC3'

@app.route('/')
def index():
    return "Welcome to the Question Generator API. Use /generate to generate questions and /export to export questions."

@app.route('/generate', methods=['POST'])
def generate_questions():
    data = request.get_json()
    
    topic = data.get('topic', '')
    num_questions = data.get('num_questions', 1)
    difficulty = data.get('difficulty', '')
    language = data.get('language', '')
    
    # Filter the questions based on the input parameters
    filtered_df = questions_df[
        (questions_df['question_data'].str.contains(topic, case=False, na=False)) &
        (questions_df['manual_difficulty'].str.contains(difficulty, case=False, na=False)) &
        (questions_df['multilanguage'].str.contains(language, case=False, na=False))
    ]
    
    # If there are not enough questions, return all available questions
    if len(filtered_df) < num_questions:
        selected_questions = filtered_df.to_dict(orient='records')
    else:
        selected_questions = filtered_df.sample(n=num_questions).to_dict(orient='records')
    
    return jsonify(selected_questions)

@app.route('/export', methods=['POST'])
def export_questions():
    data = request.get_json()
    questions = data.get('questions', [])
    
    # Export the questions to a CSV file
    export_df = pd.DataFrame(questions)
    export_csv_path = 'exported_questions.csv'
    export_df.to_csv(export_csv_path, index=False)
    
    return jsonify({"message": f"Questions exported to {export_csv_path}"})

if __name__ == '__main__':
    app.run(debug=True)


import os
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
import openai

app = Flask(__name__)

# Load your CSV file
csv_file_path = "D://Iamneo//Parsed_Questions (1).csv"
questions_df = pd.read_csv(csv_file_path)

# Configure OpenAI
openai.api_key = 'sk-proj-1S7rLZqc75o9Ke6XChrjT3BlbkFJDSll9JeZb3yKYmui5dC3'

def generate_questions(topic, num_questions, difficulty, language):
    prompt = f"Generate {num_questions} {difficulty} questions on {topic} in {language}."
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=1000
    )
    questions = response.choices[0].text.strip().split('\n')
    return questions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    topic = data.get('topic')
    num_questions = data.get('num_questions')
    difficulty = data.get('difficulty')
    language = data.get('language')
    
    questions = generate_questions(topic, num_questions, difficulty, language)
    
    return jsonify({'questions': questions})

@app.route('/export', methods=['POST'])
def export():
    data = request.json
    export_format = data.get('format', 'csv')
    
    if export_format == 'csv':
        export_path = 'exported_questions.csv'
        questions_df.to_csv(export_path, index=False)
    elif export_format == 'json':
        export_path = 'exported_questions.json'
        questions_df.to_json(export_path, orient='records')
    
    return send_file(export_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
    

import os
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
import openai

app = Flask(__name__)

# Configure OpenAI
openai.api_key = 'sk-proj-hejDE3KLrHyHxeW2mYmiT3BlbkFJJBFAwpTPxacFcaZhQmPe'

# Example: Test API connectivity
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Hello, how are you?"},
        {"role": "user", "content": "I'm doing great, thanks! How about you?"}
    ]
)
print(response)

# Load your CSV file
csv_file_path = "D://Iamneo//Parsed_Questions (1).csv"
questions_df = pd.read_csv(csv_file_path)

def generate_questions(topic, num_questions, difficulty, language):
    prompt = f"Generate {num_questions} {difficulty} questions on {topic} in {language}."
    response = openai.Completion.create(
        model="davinci-codex",
        prompt=prompt,
        max_tokens=1000,
        stop=None  # You can add stop conditions if needed
    )
    questions = response['choices'][0]['text'].strip().split('\n')
    return questions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    topic = data.get('topic')
    num_questions = data.get('num_questions')
    difficulty = data.get('difficulty')
    language = data.get('language')   
    questions = generate_questions(topic, num_questions, difficulty, language)   
    return jsonify({'questions': questions})
@app.route('/export', methods=['POST'])
def export():
    data = request.json
    export_format = data.get('format', 'csv')
    
    if export_format == 'csv':
        export_path = 'exported_questions.csv'
        questions_df.to_csv(export_path, index=False)
    elif export_format == 'json':
        export_path = 'exported_questions.json'
        questions_df.to_json(export_path, orient='records')   
    return send_file(export_path, as_attachment=True)
if __name__ == '__main__':
    app.run(debug=True)
# Configure OpenAI
openai.api_key = 'sk-proj-hejDE3KLrHyHxeW2mYmiT3BlbkFJJBFAwpTPxacFcaZhQmPe'   
"""