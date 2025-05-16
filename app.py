import google.generativeai as genai
import re
import os
import ast
from itertools import zip_longest
import pandas as pd
from serpapi import GoogleSearch
import faiss
from flask import session, redirect
from sentence_transformers import SentenceTransformer
import numpy as np
# cv2 is imported but not used, consider removing if truly unused
# import cv2
from flask import Flask, render_template, Response, request, jsonify, url_for
# PIL is imported but not used, consider removing if truly unused
# from PIL import Image
from dotenv import load_dotenv
import uuid
import logging
from langchain_community.document_loaders import PyPDFLoader # Updated import path for newer langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Consider using a more stable model name if available, experimental ones might change
geminimodel = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-04-17") # Updated to a common flash model
model = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)
# app.secret_key = 'dev_key_123'
# UPLOAD_FOLDER and ALLOWED_EXTENSIONS seem unused in the provided code for book processing
# They might be for a different feature (e.g., user PDF uploads)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Define base directories for organized files
PDF_DIR = "pdf_files"
INDEX_DIR = "index_files"
# Ensure these directories exist if needed for writing, though reading is the primary focus here
# os.makedirs(PDF_DIR, exist_ok=True)
# os.makedirs(INDEX_DIR, exist_ok=True)

def recommendTopics(topic):
    # Assuming this CSV remains in the root or its own specified path
    csv_path = 'upsc_real_subtopics.csv'
    index_path = 'plagarism.index' # Assuming this index is separate from book indices

    if not os.path.exists(csv_path):
        logger.error(f"Required CSV file not found: {csv_path}")
        return ["Error: Subtopics data file missing."]
    try:
        final_df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error reading CSV {csv_path}: {e}")
        return [f"Error reading subtopics data: {e}"]

    test = final_df["Subtopics"].apply(lambda text: str(text).lower())
    texts = test.to_list()

    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            query_embedding = model.encode([topic], convert_to_tensor=False)
            top_k = 3
            distances, indices = index.search(query_embedding, top_k)
            recommended_texts = [texts[i] for i in indices[0]]
            logger.info(f"Recommended texts for '{topic}': {recommended_texts}")
            return recommended_texts
        except Exception as e:
            logger.error(f"Error loading or searching index {index_path}: {e}")
            return [f"Error processing recommendations: {e}"]
    else:
        logger.warning(f"Index file {index_path} not found. Generating temporarily (will not persist across runs).")
        # Note: Generating index here might be slow and not ideal for production
        try:
            embeddings = model.encode(texts, convert_to_tensor=False)
            embedding_dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(embeddings)
            # Consider saving the index if generation is intended, maybe only if it doesn't exist
            # faiss.write_index(index, index_path)
            query_embedding = model.encode([topic], convert_to_tensor=False)
            top_k = 3
            distances, indices = index.search(query_embedding, top_k)
            recommended_texts = [texts[i] for i in indices[0]]
            logger.info(f"Recommended texts for '{topic}' (generated index): {recommended_texts}")
            return recommended_texts
        except Exception as e:
            logger.error(f"Error generating/searching index for {index_path}: {e}")
            return [f"Error generating recommendations: {e}"]


def generate_questions(topic):
    prompt = f"""You are an expert in {topic}. Your task is to generate 10 multiple-choice questions (MCQs) covering diverse topics within this subject.
    Each question must be followed by four answer choices (a, b, c, d), and the correct answer must be among them.
    Output Format:
    Q1: [Question]?
    a) [Option 1]
    b) [Option 2]
    c) [Option 3]
    d) [Option 4]
    ... (up to Q10)
    Answer List: ['Correct Option for Q1', 'Correct Option for Q2', ..., 'Correct Option for Q10'] (Correct options exactly as written in the choices above)
    """
    try:
        response = geminimodel.generate_content(prompt)
        # Added basic safety check
        if not response.parts:
             logger.error("No response parts received from Gemini model.")
             return [], 0
        text = response.text # Use .text directly
    except Exception as e:
         logger.error(f"Error generating content from Gemini: {e}")
         return [], 0

    print("-------------------------------------")
    print(text)
    print("--------------------------------------")
    # Improved regex and parsing robustness
    questions = re.findall(r'Q\d+:\s*(.*?)\?', text, re.IGNORECASE | re.DOTALL)
    # Ensure options capture multi-line possibilities within an option, stopping at the next letter or Answer List
    options_raw = re.findall(r'a\)\s*(.*?)\s*b\)\s*(.*?)\s*c\)\s*(.*?)\s*d\)\s*(.*?)(?=\nQ\d+|\nAnswer List:|\Z)', text, re.IGNORECASE | re.DOTALL)
    options = [[opt.strip() for opt in group] for group in options_raw] # Clean whitespace

    match = re.search(r"Answer List:\s*(\[.*?\])", text, re.IGNORECASE | re.DOTALL)
    correct_answers = []
    if match:
        try:
            # Use literal_eval carefully, ensure input is trusted or sanitize
            raw_list_str = match.group(1)
            correct_answers = ast.literal_eval(raw_list_str)
            correct_answers = [str(ans).strip() for ans in correct_answers] # Ensure strings and strip
        except (ValueError, SyntaxError) as e:
            logger.error(f"Error parsing Answer List: {e}. Raw string: '{match.group(1)}'")
            correct_answers = [] # Fallback to empty list

    num_questions_found = len(questions)
    num_options_found = len(options)
    num_answers_found = len(correct_answers)

    if not (num_questions_found == num_options_found == num_answers_found):
         logger.warning(f"Mismatch in parsed questions ({num_questions_found}), options ({num_options_found}), and answers ({num_answers_found}). Check LLM output format.")
         # Attempt to align based on the minimum count
         min_count = min(num_questions_found, num_options_found, num_answers_found)
         questions = questions[:min_count]
         options = options[:min_count]
         correct_answers = correct_answers[:min_count]
         count = min_count
    else:
        count = num_questions_found

    logger.debug(f"Generated {count} questions.")
    logger.debug(f"Correct answers parsed: {correct_answers}")

    question_data = []
    for i in range(count):
        q = questions[i]
        opt = options[i]
        ans = correct_answers[i]

        # Basic validation: ensure the listed correct answer is one of the options
        if ans not in opt:
            logger.warning(f"Correct answer '{ans}' for Q{i+1} not found in options {opt}. Defaulting to first option or handling as error might be needed.")
            # Decide on fallback: Use first option? Mark as invalid? Skip question?
            # Using first option for now, but this indicates an LLM output issue.
            corrected_answer = opt[0] if opt else ""
        else:
            corrected_answer = ans

        question_data.append({
            "question": q,
            "options": opt,
            "correct": corrected_answer
        })

    return question_data, count


def get_subtopic(question):
    prompt = f"Analyze the following question and classify it into a single, concise subtopic or medical specialty. Examples: Cardiology, Infectious Disease, Pharmacology, Renal Physiology, etc.\n\nQuestion: {question}\n\nSubtopic:"
    try:
        response = geminimodel.generate_content(prompt)
        if response and response.parts:
            # Added stripping of potential markdown like **Subtopic**
            subtopic = response.text.strip().replace('*', '').replace('`', '')
            logger.debug(f"Subtopic for question '{question[:50]}...': {subtopic}")
            return subtopic
        else:
            logger.warning(f"Could not determine subtopic for question: {question[:50]}...")
            return "Unknown Subtopic"
    except Exception as e:
        logger.error(f"Error getting subtopic from Gemini: {e}")
        return "Subtopic Error"


# --- Updated BOOKS List with os.path.join ---
BOOKS = [
    {"name": "Harrisons Rheumatology",
     "book_path": os.path.join(PDF_DIR, "Harrisons_Rheumatology.pdf"),
     "embeddings_path": os.path.join(INDEX_DIR, "Harrisons_Rheumatology.index")},
    {"name": "Harrisons Nephrology And Acid Base",
     "book_path": os.path.join(PDF_DIR, "Harrisons_Nephrology_and_Acid_Base.pdf"),
     "embeddings_path": os.path.join(INDEX_DIR, "Harrisons_Nephrology_and_Acid_Base.index")},
    {"name": "Harrisons Manual Of Medicine",
     "book_path": os.path.join(PDF_DIR, "Harrisons_Manual_Of_Medicine.pdf"),
     "embeddings_path": os.path.join(INDEX_DIR, "Harrisons_Manual_Of_Medicine.index")},
    {"name": "Harrisons Manual Of Oncology",
     "book_path": os.path.join(PDF_DIR, "Harrisons_Manual_of_Oncology.pdf"),
     "embeddings_path": os.path.join(INDEX_DIR, "Harrisons_Manual_of_Oncology.index")},
    {"name": "Harrisons Pulmonary And Critical Care Medicine",
     "book_path": os.path.join(PDF_DIR, "Harrisons_Pulmonary_and_Critical_Care_Medicine.pdf"),
     "embeddings_path": os.path.join(INDEX_DIR, "Harrisons_Pulmonary_and_Critical_Care_Medicine.index")},
    {"name": "Harrisons Cardio Vascular Medicine ",
     "book_path": os.path.join(PDF_DIR, "harrisons_cardiovascular_medicine.pdf"),
     "embeddings_path": os.path.join(INDEX_DIR, "harrisons_cardiovascular_medicine.index")},
    {"name": "Harrisons Endocrinology",
     "book_path": os.path.join(PDF_DIR, "harrisons_endocrinology.pdf"),
     "embeddings_path": os.path.join(INDEX_DIR, "harrisons_endocrinology.index")},
    {"name": "Harrisons Gastroenterology And Hepatology",
     "book_path": os.path.join(PDF_DIR, "harrisons_gastroenterology_and_hepatology.pdf"),
     "embeddings_path": os.path.join(INDEX_DIR, "harrisons_gastroenterology_and_hepatology.index")},
    {"name": "Harrisons Hematology And Oncology",
     "book_path": os.path.join(PDF_DIR, "harrisons_hematology_and_oncology.pdf"),
     "embeddings_path": os.path.join(INDEX_DIR, "harrisons_hematology_and_oncology.index")},
    # Duplicate entries removed for clarity, assuming they were accidental
    # {"name": "Harrisons Manual Of Medicine", ...},
    # {"name": "Harrisons Nephrology And Acid Base", ...},
    {"name": "Harrisons Neurology In Clinical Medicine",
     "book_path": os.path.join(PDF_DIR, "Harrisons_neurology_in_clinical_medicine.pdf"),
     "embeddings_path": os.path.join(INDEX_DIR, "Harrisons_neurology_in_clinical_medicine.index")},
    {"name":"PathologyRobbins",
     "book_path": os.path.join(PDF_DIR,"PathologyRobbins7ed.pdf"),
     "embeddings_path": os.path.join(INDEX_DIR,"PathologyRobbins7ed.index")},
    {"name":"KD Tripathi Essentials of Medical Pharmacology",
     "book_path": os.path.join(PDF_DIR,"KD_Tripathi_Essentials_of_Medical_PharmacologyUnitedVRG_2013.pdf"),
     # Corrected embedding path assuming it should match the book name pattern
     "embeddings_path": os.path.join(INDEX_DIR,"KD_Tripathi_Essentials_of_Medical_PharmacologyUnitedVRG_2013.index")},
    {"name":"Harpers Illustrated Biochemistry",
     "book_path": os.path.join(PDF_DIR,"Harpers_Illustrated_Biochemistry_Thirty_Second_Edition_2023.pdf"),
     "embeddings_path": os.path.join(INDEX_DIR,"Harpers_Illustrated_Biochemistry_Thirty_Second_Edition_2023.index")},
    {"name":"Gray Anatomy",
     "book_path": os.path.join(PDF_DIR,"Grays_Anatomy-41_E.pdf"),
     "embeddings_path": os.path.join(INDEX_DIR,"Grays_Anatomy-41_E.index")}
]
# --- End of Updated BOOKS List ---


@app.route('/book_test')
def books_name():
    return render_template('book_dropdown.html', books=BOOKS)

@app.route('/get_paths', methods=['POST'])
def get_paths():
    book_name = request.form.get('book_name')
    query = request.form.get('query', '') # Allow user to specify query later
    if not query:
         # Provide a default query if none is given, or return an error/prompt
         query = "Transdermal therapeutic systems (TTS)" # Example default
         logger.warning("No query provided, using default: " + query)

    logger.info(f"Received book name: {book_name}, Query: {query}")

    selected_book = next((book for book in BOOKS if book["name"] == book_name), None)

    if not selected_book:
        logger.error(f"Book name '{book_name}' not found in configuration.")
        # Return an error message to the user
        return "Error: Selected book not found.", 404

    # --- Paths are now correctly taken from the selected_book dictionary ---
    book_path = selected_book["book_path"]
    book_embeddings_path = selected_book["embeddings_path"]
    logger.info(f"Using PDF path: {book_path}")
    logger.info(f"Using Index path: {book_embeddings_path}")

    # --- Helper functions defined within the route ---
    # It might be cleaner to define these outside the route if they don't strictly depend on route variables beyond paths/chunks
    def load_pdf_and_split(pdf_path):
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        try:
            # Using updated import: langchain_community.document_loaders
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            # Consider adjusting chunk_size/overlap based on content and model limits
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150) # Slightly more overlap
            chunks = splitter.split_documents(documents)
            logger.info(f"Loaded and split {pdf_path} into {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logger.error(f"Error loading/splitting PDF {pdf_path}: {e}")
            raise # Re-raise the exception to be caught by the main route handler

    def generate_embeddings(chunks, index_save_path):
        # This function should ideally only be called if the index doesn't exist and needs creation
        try:
            # Re-initialize model within function or ensure it's thread-safe if global
            local_model = SentenceTransformer('all-MiniLM-L6-v2')
            texts = [chunk.page_content for chunk in chunks]
            if not texts:
                logger.warning("No text found in chunks to generate embeddings.")
                return None
            logger.info(f"Generating embeddings for {len(texts)} text chunks...")
            embeddings = local_model.encode(texts, convert_to_numpy=True, show_progress_bar=True) # Show progress
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            # Ensure the directory exists before writing
            os.makedirs(os.path.dirname(index_save_path), exist_ok=True)
            faiss.write_index(index, index_save_path)
            logger.info(f"Generated and saved embeddings to {index_save_path}")
            return index
        except Exception as e:
            logger.error(f"Error generating/saving embeddings to {index_save_path}: {e}")
            raise

    def load_embeddings(index_path):
        if not os.path.exists(index_path):
             logger.error(f"Embeddings index file not found: {index_path}")
             raise FileNotFoundError(f"Embeddings index file not found: {index_path}")
        try:
            index = faiss.read_index(index_path)
            logger.info(f"Successfully loaded embeddings from {index_path}")
            return index
        except Exception as e:
            logger.error(f"Error loading embeddings index {index_path}: {e}")
            raise

    def retrieve_similar_chunks(index, query, chunks, top_k=3):
        try:
            # Re-initialize model or ensure thread safety
            local_model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = local_model.encode([query], convert_to_numpy=True)
            logger.info(f"Searching index for query: '{query}'")
            distances, indices = index.search(query_embedding, top_k)
            # Filter out potential invalid indices (if top_k > number of items in index)
            valid_indices = [idx for idx in indices[0] if 0 <= idx < len(chunks)]
            similar_chunks = [(chunks[idx], distances[0][i]) for i, idx in enumerate(valid_indices)]
            logger.info(f"Retrieved {len(similar_chunks)} similar chunks.")
            return similar_chunks
        except Exception as e:
            logger.error(f"Error retrieving similar chunks: {e}")
            raise

    def generate_book_questions(retrieved_text, query): # Renamed to avoid conflict
        # Combine retrieved texts into a single context string
        context = "\n\n".join(retrieved_text)
        if not context:
             logger.warning("No context provided to generate questions.")
             return []

        prompt = f"""Based *only* on the following text excerpts related to '{query}', generate exactly 5 challenging multiple-choice questions (MCQs). Each question must test in-depth knowledge found within the provided text.

        Provided Text Excerpts:
        ---
        {context}
        ---

        Instructions:
        1. Create 5 MCQs with four options (a, b, c, d) each.
        2. Ensure the questions and options are derived *directly* from the provided text.
        3. The correct answer must be one of the four options.
        4. Format the output precisely as shown below.

        Output Format:
        Q1: [Question 1]?
        a) [Option 1a]
        b) [Option 1b]
        c) [Option 1c]
        d) [Option 1d]

        Q2: [Question 2]?
        a) [Option 2a]
        b) [Option 2b]
        c) [Option 2c]
        d) [Option 2d]

        Q3: [Question 3]?
        a) [Option 3a]
        b) [Option 3b]
        c) [Option 3c]
        d) [Option 3d]

        Q4: [Question 4]?
        a) [Option 4a]
        b) [Option 4b]
        c) [Option 4c]
        d) [Option 4d]

        Q5: [Question 5]?
        a) [Option 5a]
        b) [Option 5b]
        c) [Option 5c]
        d) [Option 5d]

        Answer List: ['Correct Option for Q1', 'Correct Option for Q2', 'Correct Option for Q3', 'Correct Option for Q4', 'Correct Option for Q5']
        """
        try:
            logger.info("Generating book-based questions from retrieved text...")
            response = geminimodel.generate_content(prompt)
            if not response.parts:
                logger.error("No response parts received from Gemini for book questions.")
                return []
            text = response.text
        except Exception as e:
            logger.error(f"Error generating book questions from Gemini: {e}")
            return []
        print("--------------------------------------------------")
        print(text)
        print("--------------------------------------------------")
        # --- Parsing logic similar to generate_questions, adapted for 5 questions ---
        questions = re.findall(r'Q\d+:\s*(.*?)\?', text, re.IGNORECASE | re.DOTALL)[:5]
        options_raw = re.findall(r'a\)\s*(.*?)\s*b\)\s*(.*?)\s*c\)\s*(.*?)\s*d\)\s*(.*?)(?=\nQ\d+|\nAnswer List:|\Z)', text, re.IGNORECASE | re.DOTALL)[:5]
        options = [[opt.strip() for opt in group] for group in options_raw]

        match = re.search(r"Answer List:\s*(\[.*?\])", text, re.IGNORECASE | re.DOTALL)
        correct_answers = []
        if match:
            try:
                raw_list_str = match.group(1)
                correct_answers = ast.literal_eval(raw_list_str)
                correct_answers = [str(ans).strip() for ans in correct_answers][:5] # Limit to 5
            except (ValueError, SyntaxError) as e:
                logger.error(f"Error parsing Answer List for book questions: {e}. Raw: '{match.group(1)}'")
                correct_answers = []

        num_q = len(questions)
        num_o = len(options)
        num_a = len(correct_answers)

        if not (num_q == num_o == num_a == 5):
             logger.warning(f"Book questions: Expected 5 Q/O/A, found Q:{num_q}, O:{num_o}, A:{num_a}. Truncating/padding might occur.")
             # Adjust lists to the minimum common length, up to 5
             min_len = min(num_q, num_o, num_a, 5)
             questions = questions[:min_len]
             options = options[:min_len]
             correct_answers = correct_answers[:min_len]

        question_data = []
        for i in range(len(questions)): # Iterate based on available aligned data
            q = questions[i]
            opt = options[i]
            ans = correct_answers[i]

            if ans not in opt:
                logger.warning(f"Book Q{i+1}: Correct answer '{ans}' not in options {opt}. Defaulting to first.")
                corrected_answer = opt[0] if opt else ""
            else:
                corrected_answer = ans

            question_data.append({
                "question": q,
                "options": opt,
                "correct": corrected_answer
            })
        logger.info(f"Generated {len(question_data)} book-based questions.")
        return question_data
    # --- End of helper function definitions ---

    try:
        # 1. Load and split PDF
        chunks = load_pdf_and_split(book_path)
        if not chunks:
            return "Error: Could not process PDF content.", 500

        # 2. Load or Generate Embeddings
        index = None
        # --- Corrected logic: check for index file FIRST ---
        if os.path.exists(book_embeddings_path):
            logger.info("Embeddings index found. Loading...")
            index = load_embeddings(book_embeddings_path)
        else:
            # This part should ideally be an offline process, but included as per original logic
            logger.warning(f"Embeddings index not found at {book_embeddings_path}. Generating and saving... This may take time.")
            # Call generate_embeddings which now saves the index
            index = generate_embeddings(chunks, book_embeddings_path)

        if index is None:
             # Handle case where index loading and generation failed
             return "Error: Could not load or generate embeddings index.", 500


        # 3. Retrieve relevant chunks based on the query
        similar_chunks_data = retrieve_similar_chunks(index, query, chunks, top_k=5) # Retrieve more chunks for better context
        retrieved_texts = [chunk.page_content for chunk, distance in similar_chunks_data]

        if not retrieved_texts:
            logger.warning(f"No relevant text chunks found for query '{query}' in {book_name}.")
            # Return message indicating no relevant content found
            return f"Could not find relevant information for '{query}' in the selected book.", 200 # Or 404?

        # 4. Generate questions based on retrieved text
        question_data = generate_book_questions(retrieved_texts, query)

        if not question_data:
            # Handle case where question generation failed or returned empty
            logger.warning("Question generation based on retrieved text returned no questions.")
            return f"Found relevant information for '{query}', but could not generate questions.", 200

        # 5. Render the questions
        # Pass the query to the template as well
        return render_template("books_questions.html", questions_data=question_data, book_name=book_name, query=query)

    except FileNotFoundError as e:
        logger.error(f"File not found error in /get_paths: {e}")
        return f"Error: A required file was not found ({e}). Please check server configuration.", 500
    except Exception as e:
        logger.error(f"An unexpected error occurred in /get_paths: {e}", exc_info=True) # Log traceback
        return "An unexpected error occurred while processing your request.", 500


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        topic = request.form.get('topic', 'General Knowledge').strip() # Add default and strip whitespace
        if not topic:
             topic = 'General Knowledge' # Ensure topic is not empty
             logger.warning("Received empty topic, defaulting to 'General Knowledge'.")
        logger.info(f"Generating quiz for topic: {topic}")
        questions, count = generate_questions(topic)
        if not questions:
             # Handle case where question generation fails
             logger.error(f"Failed to generate any questions for topic: {topic}")
             # Redirect back or show error message
             return render_template('index.html', error="Could not generate questions for this topic. Please try again or choose a different topic.")
        # Pass topic and count to quiz template
        return render_template('quiz.html', questions=questions, topic=topic, count=count)
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    logger.info("Processing quiz submission.")
    try:
        # Safely evaluate the questions string from the form
        questions_str = request.form.get('questions')
        if not questions_str:
             logger.error("No questions data received in submission.")
             return "Error: Missing question data in submission.", 400
        questions = ast.literal_eval(questions_str)

        if not isinstance(questions, list):
             logger.error("Invalid format for questions data received.")
             return "Error: Invalid question data format.", 400

    except (ValueError, SyntaxError) as e:
        logger.error(f"Error parsing questions data from form: {e}")
        return "Error processing submission data.", 400
    except Exception as e:
         logger.error(f"Unexpected error processing questions data: {e}", exc_info=True)
         return "Server error during submission processing.", 500

    weak_questions_details = [] # Store dicts with question, selected, correct
    weak_topics = []
    score = 0
    total = len(questions)

    for i, question in enumerate(questions):
        selected = request.form.get(f'question{i}', '').strip()
        # Ensure 'correct' key exists and handle potential issues
        correct_answer = question.get('correct', '').strip()
        question_text = question.get('question', 'Unknown Question')
        options = question.get('options', [])

        if not correct_answer:
             logger.warning(f"Missing correct answer for question index {i}: {question_text}")
             # Decide how to handle: skip scoring? mark as incorrect?
             continue # Skipping this question for scoring

        logger.debug(f"Q{i}: Selected='{selected}', Correct='{correct_answer}'")
        if selected == correct_answer:
            score += 1
        else:
            # Store more details about the wrongly answered question
            weak_questions_details.append({
                "question": question_text,
                "options": options,
                "selected": selected,
                "correct": correct_answer
            })
            # Get subtopic for the wrongly answered question
            weak_topics.append(get_subtopic(question_text))

    # --- Generate Explanations for Incorrect Answers ---
    explanation_list = []
    logger.info(f"Generating explanations for {len(weak_questions_details)} incorrect answers.")
    for detail in weak_questions_details:
        # Construct a more specific prompt for explanation
        prompt = f"""The user incorrectly answered the following multiple-choice question:

        Question: {detail['question']}
        Options:
        a) {detail['options'][0] if len(detail['options']) > 0 else 'N/A'}
        b) {detail['options'][1] if len(detail['options']) > 1 else 'N/A'}
        c) {detail['options'][2] if len(detail['options']) > 2 else 'N/A'}
        d) {detail['options'][3] if len(detail['options']) > 3 else 'N/A'}

        User's incorrect answer: {detail['selected']}
        Correct answer: {detail['correct']}

        Please provide a concise and clear explanation clarifying why '{detail['correct']}' is the correct answer and potentially why '{detail['selected']}' (if provided and different) might be incorrect or less suitable. Focus on the core concept being tested. Keep the explanation brief (2-3 sentences).
        """
        try:
            response = geminimodel.generate_content(prompt)
            explanation = response.text.strip() if response.parts else "Could not generate explanation."
            explanation_list.append(explanation)
        except Exception as e:
            logger.error(f"Error generating explanation for question '{detail['question'][:50]}...': {e}")
            explanation_list.append("Error generating explanation.")
        # Optional: Add a small delay to avoid rate limits if generating many explanations
        # time.sleep(0.5)

    # Combine weak questions (text only) and their explanations
    question_explanations = list(zip([d['question'] for d in weak_questions_details], explanation_list))

    # --- Optional: Fetch articles using SERP API for weak topics ---
    # Be mindful of API costs and rate limits
    articles_list = []
    USE_SERP_API = False # Set to True to enable SERP API calls
    if USE_SERP_API and weak_topics:
         logger.info("Fetching related articles using SERP API...")
         # Use unique weak topics to avoid redundant searches
         unique_weak_topics = list(set(weak_topics) - {"Unknown Subtopic", "Subtopic Error"}) # Exclude generic/error topics
         for weak_topic in unique_weak_topics[:3]: # Limit number of searches
             logger.debug(f"Searching SERP API for: {weak_topic}")
             try:
                 params = {
                     "q": f"{weak_topic} learning resources OR explanation", # More specific query
                     "hl": "en",
                     "api_key": os.getenv("SERP_API_KEY")
                 }
                 if not params["api_key"]:
                      logger.warning("SERP_API_KEY not found in environment variables. Skipping article search.")
                      break # Stop trying if key is missing

                 search = GoogleSearch(params)
                 results = search.get_dict()

                 if "organic_results" in results:
                     for result in results["organic_results"][:2]: # Get top 2 links per topic
                         articles_list.append({
                             "topic": weak_topic, # Changed from 'question' to 'topic'
                             "title": result.get('title', 'No Title'),
                             "link": result.get('link', '#'),
                             "snippet": result.get('snippet', '')
                         })
                 # Optional delay between API calls
                 # time.sleep(1)
             except Exception as e:
                 logger.error(f"Error fetching SERP API results for '{weak_topic}': {e}")


    logger.info(f"Quiz finished. Score: {score}/{total}")
    # Pass unique weak topics to results page as well
    unique_weak_topics_display = list(set(weak_topics) - {"Unknown Subtopic", "Subtopic Error"})
    return render_template('result.html',
                           score=score,
                           total=total,
                           weak_topics=unique_weak_topics_display,
                           articles=articles_list, # Pass fetched articles
                           question_explanations=question_explanations)


@app.route('/recommendations', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        topic = request.form.get('topic', '').strip()
        if not topic:
             # Handle empty topic input
             return render_template('recommendationsForm.html', error="Please enter a topic.")
        logger.info(f"Getting recommendations for topic: {topic}")
        recommended_topics = recommendTopics(topic) # Uses the separate index
        return render_template('recommend.html', recommendedTopics=recommended_topics, topic=topic)
    return render_template('recommendationsForm.html')


# --- Chat Functionality ---
chat_sessions = {} # Store active chat objects {chat_id: chat_object}

@app.route('/chatbuddy', methods=["GET", "POST"])
def chatting():
    if request.method == "POST":
        topic = request.form.get("topic", "General Knowledge").strip()
        if not topic:
             topic = "General Knowledge"
        chat_id = str(uuid.uuid4()) # Generate a unique ID for this chat session
        try:
            # Start a new chat session with the LLM
            chat_sessions[chat_id] = geminimodel.start_chat(history=[]) # Start with empty history
            logger.info(f"Started new chat session {chat_id} for topic: {topic}")
        except Exception as e:
            logger.error(f"Failed to start chat session with Gemini: {e}")
            return render_template("chatting_template.html", error="Could not start chat session. Please try again later.")

        # Store chat details in Flask session
        session["topic"] = topic
        session["chat_id"] = chat_id
        # History will now be managed within the chat_object, but we can keep a parallel list in session for display if needed
        session["display_history"] = [] # Separate list for rendering

        return redirect(url_for("chat")) # Redirect to the chat interface
    return render_template("chatting_template.html") # Show form to start chat

@app.route("/chat", methods=["GET", "POST"])
def chat():
    # Check if chat session details exist in Flask session
    if "topic" not in session or "chat_id" not in session:
        logger.warning("Chat accessed without active session details. Redirecting to start.")
        return redirect(url_for("chatting"))

    topic = session["topic"]
    chat_id = session["chat_id"]

    # Retrieve the actual chat object using the chat_id
    chat_obj = chat_sessions.get(chat_id)

    # If chat object doesn't exist (e.g., server restarted, invalid ID), redirect
    if not chat_obj:
        logger.warning(f"Chat object for session {chat_id} not found. Clearing session and redirecting.")
        session.clear() # Clear potentially stale session data
        return redirect(url_for("chatting"))

    # Use the display history stored in the session
    display_history = session.get("display_history", [])

    if request.method == "POST":
        user_input = request.form.get("message", "").strip()
        if user_input: # Process only if message is not empty
             if user_input.lower() == "exit":
                 logger.info(f"User requested exit for chat session {chat_id}.")
                 return redirect(url_for("reset")) # Go to reset route to clear session

             # Construct prompt for the LLM
             # System prompts or context can be added here or during start_chat
             prompt = f"""You are a helpful assistant knowledgeable in {topic}.
Please answer the following question clearly and concisely:
{user_input}"""

             logger.debug(f"Sending message to Gemini for chat {chat_id}: {user_input}")
             try:
                 # Send message using the retrieved chat object
                 # The chat object maintains the history internally
                 response = chat_obj.send_message(prompt)
                 bot_reply = response.text.strip()
                 logger.debug(f"Received reply from Gemini for chat {chat_id}: {bot_reply[:100]}...")

                 # Update the display history in the session
                 display_history.append(("You", user_input))
                 display_history.append(("Gemini", bot_reply))
                 session["display_history"] = display_history # Save updated history back to session

                 # No need to manually manage chat_obj.history if using start_chat() correctly,
                 # but ensure the display history is updated.

             except Exception as e:
                 logger.error(f"Error sending message or receiving reply in chat {chat_id}: {e}")
                 # Optionally inform the user in the chat interface
                 display_history.append(("System", "Sorry, there was an error processing your message."))
                 session["display_history"] = display_history


    # Render the chat template with the current topic and display history
    return render_template("chat.html", topic=topic, history=display_history)


@app.route("/reset")
def reset():
    chat_id = session.get("chat_id")
    if chat_id and chat_id in chat_sessions:
        # Optional: Clean up the chat object if necessary (though Python's GC might handle it)
        del chat_sessions[chat_id]
        logger.info(f"Removed chat session {chat_id} from active sessions.")
    session.clear() # Clear the Flask session data (topic, chat_id, history)
    logger.info("Chat session cleared. Redirecting to chat start.")
    return redirect(url_for("chatting"))


if __name__ == '__main__':
    # Ensure necessary directories exist before starting the app
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) # For the unused upload feature
    # Turn off reloader in production, debug=False
    app.run(debug=True, use_reloader=True) # use_reloader=True is default with debug=True