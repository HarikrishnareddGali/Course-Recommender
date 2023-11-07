from transformers import pipeline
import pandas as pd
from transformers import DistilBertTokenizer
import os
import spacy
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Check if processed data exists
CACHE_FILE = "processed_data.parquet"

if os.path.exists(CACHE_FILE):
    # Load from cache
    grouped = pd.read_parquet(CACHE_FILE)
else:
    # Read data from the Excel file
    df = pd.read_excel("output_file.xlsx")


    def get_average_score(row):
        total_students = row['A'] + row['A-'] + row['B+'] + row['B'] + row['B-'] + row['C+'] + row['C'] + row['C-'] + \
                         row[
                             'D+'] + row['D'] + row['F']

        if total_students == 0:
            return 0  # or any other value you deem appropriate for this case

        average_score = (row['A'] * 4 + row['A-'] * 3.7 + row['B+'] * 3.3 + row['B'] * 3 + row['B-'] * 2.7 + row[
            'C+'] * 2.3 + row['C'] * 2 + row['C-'] * 1.7 + row['D+'] * 1.3 + row['D'] * 1) / total_students
        return average_score


    # Group by course name and instructor, then aggregate the data
    grouped = df.groupby(['Course', 'Instructor', 'Department']).agg({
        'A': 'sum', 'A-': 'sum', 'B+': 'sum', 'B': 'sum', 'B-': 'sum',
        'C+': 'sum', 'C': 'sum', 'C-': 'sum', 'D+': 'sum', 'D': 'sum', 'F': 'sum',
        'Good Comments': lambda x: ' '.join(map(str, x)),
        'Bad Comments': lambda x: ' '.join(map(str, x)),
        'Section': lambda x: list(map(str, x.unique()))
    }).reset_index()

    grouped['Average Score'] = grouped.apply(get_average_score, axis=1)

    # Create a sentiment analysis pipeline
    nlp = pipeline("sentiment-analysis")

    MAX_TOKENS = 500


    def truncate_text(text):
        tokens = tokenizer.tokenize(text)
        if len(tokens) > MAX_TOKENS:
            tokens = tokens[:MAX_TOKENS]
        return tokenizer.convert_tokens_to_string(tokens)


    def get_sentiment_score(text):
        # Truncate or split the text if it's too long
        text = truncate_text(text)
        print(f"Token count: {len(tokenizer.tokenize(text))}")
        result = nlp(text)
        sentiment = result[0]['label']
        confidence = result[0]['score']

        if sentiment == "POSITIVE":
            return confidence * 10  # Scaled to match -10 to 10
        else:
            return -confidence * 10  # Scaled to match -10 to 10


    grouped['Sentiment Score'] = grouped['Good Comments'].apply(get_sentiment_score) - grouped['Bad Comments'].apply(
        get_sentiment_score)
    # After all processing, save to cache
    grouped.to_parquet(CACHE_FILE)

# Normalize Average Score to [0, 1]
max_avg_score = grouped['Average Score'].max()
min_avg_score = grouped['Average Score'].min()
grouped['Normalized Avg Score'] = (grouped['Average Score'] - min_avg_score) / (max_avg_score - min_avg_score)

# Normalize Sentiment Score to [0, 1]
max_sentiment_score = grouped['Sentiment Score'].max()
min_sentiment_score = grouped['Sentiment Score'].min()
grouped['Normalized Sentiment Score'] = (grouped['Sentiment Score'] - min_sentiment_score) / (
        max_sentiment_score - min_sentiment_score)

# Define weights (adjust as needed)
weight_avg_score = 0.5
weight_sentiment_score = 0.5

# Calculate combined metric
grouped['Combined Metric'] = (weight_avg_score * grouped['Normalized Avg Score']) + (
        weight_sentiment_score * grouped['Normalized Sentiment Score'])

# Categorize based on the combined metric
lower_percentile = grouped['Combined Metric'].quantile(0.33)
upper_percentile = grouped['Combined Metric'].quantile(0.66)


def plot_normalized_average_scores():
    plt.figure(figsize=(10, 6))

    # Plotting the histogram
    plt.hist(grouped['Normalized Avg Score'], bins=30, color='green', edgecolor='black')

    # Adding the title and labels
    plt.title('Distribution of Normalized AVG Scores')
    plt.xlabel('Normalized Avg Score')
    plt.ylabel('Number of Courses')

    # Displaying the plot
    plt.show()


plot_normalized_average_scores()


def course_difficulty(row):
    if row['Combined Metric'] <= lower_percentile:
        return "Hard"
    elif lower_percentile < row['Combined Metric'] <= upper_percentile:
        return "Medium Hard"
    else:
        return "Easy"


grouped['Difficulty'] = grouped.apply(course_difficulty, axis=1)


# Use the helper function to determine sentiment description
def get_sentiment_description(score):
    if score > 7:
        return "This course has received overwhelmingly positive feedback."
    elif 2 < score <= 7:
        return "This course has received mostly positive feedback."
    elif -2 < score <= 2:
        return "The feedback for this course has been mixed."
    else:
        return "This course has received mostly negative feedback."


def get_courses_based_on_difficulty(difficulty, department=None, num_courses=1):
    if department:
        courses = grouped[
            (grouped['Difficulty'] == difficulty) & (grouped['Department'].str.contains(department, case=False))]
    else:
        courses = grouped[grouped['Difficulty'] == difficulty]

    if len(courses) < num_courses:  # Handle the case of not enough courses
        num_courses = len(courses)
    recommended_courses = courses.sample(n=num_courses) if num_courses > 0 else courses

    return [(row['Course'], row['Instructor'], row['Section'], row['Course Overview']) for _, row in
            recommended_courses.iterrows()]


def generate_course_overview(row):
    sentiment_description = get_sentiment_description(row['Sentiment Score'])

    good_comments = str(row['Good Comments']) if pd.notna(row['Good Comments']) else "No positive feedback provided."
    bad_comments = str(row['Bad Comments']) if pd.notna(row['Bad Comments']) else "No negative feedback mentioned."

    good_comments_summary = f"Positive Feedback: {good_comments}"
    bad_comments_summary = f"Negative Feedback: {bad_comments}"

    return f"{sentiment_description}"


grouped['Course Overview'] = grouped.apply(generate_course_overview, axis=1)

nlp = spacy.load("en_core_web_sm")


# Usage:
def parse_user_input_spacy(text, departments):
    """Extracts the number of courses, their difficulties, and department using NLP."""
    doc = nlp(text)

    # Initialize result
    result = []

    # Initialize temporary variables to capture information
    current_num = None
    current_difficulty = None
    department_name = None

    # Iterate over tokens in the document
    for token in doc:
        # Capture numbers (for the number of courses)
        if token.pos_ == "NUM":
            current_num = int(token.text)

        # Capture difficulty level
        if token.text in ["hard", "easy", "medium", "medium hard"]:
            if current_difficulty:
                current_difficulty += " " + token.text
            else:
                current_difficulty = token.text

    # Capture department name based on noun chunks and match with available departments
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        for department in departments:
            if department.lower() in chunk_text:
                department_name = department
                break

    # If no department is identified from noun chunks, set it to 'None'
    if not department_name:
        department_name = None

    # Capture department with the rest of the data
    if current_num is not None and current_difficulty is not None:
        result.append((current_num, current_difficulty, department_name))

    return result


def map_to_defined_difficulties(difficulty):
    """Maps user input difficulty to one of the predefined difficulties."""
    difficulty_mapping = {
        "hard": "Hard",
        "easy": "Easy",
        "medium": "Medium Hard",
        "medium hard": "Medium Hard"
    }
    return difficulty_mapping.get(difficulty.lower(), "Medium Hard")  # Default to "Medium Hard" if not found


app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    user_input = request.json.get("message", "")
    available_departments = grouped['Department'].unique().tolist()  # Get all unique department names
    parsed_data = parse_user_input_spacy(user_input, available_departments)

    # Debug line for your reference
    print(f"Debug: Parsed data - {parsed_data}")

    parsed_data = [(num, map_to_defined_difficulties(diff), dept) for num, diff, dept in parsed_data]
    responses = []

    if parsed_data:
        for num_courses, difficulty, department in parsed_data:
            recommended_courses = get_courses_based_on_difficulty(difficulty, department, num_courses)
            for course, instructor, sections, overview in recommended_courses:
                course_code = sections[0].split('-')[0] if sections else "Unknown Code"
                response = f"Course: {course_code} Course Name: {course}, Instructor: {instructor}\nOverview: {overview}\n"
                responses.append(response)
    else:
        response = "Sorry, I couldn't understand. Please use the format 'Help me to choose X [difficulty] subjects and Y [difficulty] subject'."
        responses.append(response)

    return jsonify({"responses": responses})


if __name__ == "__main__":
    app.run(debug=True)
