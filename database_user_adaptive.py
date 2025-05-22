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
from flask import Flask, render_template, Response, request, jsonify, url_for
from dotenv import load_dotenv
load_dotenv()

import uuid
import logging
from langchain_community.document_loaders import PyPDFLoader # Updated import path for newer langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

import json
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate # For Alembic migrations

# --- 4. Import db and models from your models.py file ---
# Assuming models.py is in the same directory
from models import db, User, QuizAttempt, TopicPerformance, SubtopicMastery, AnsweredQuestion

USER_DATA_DIR = "user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)

# def load_user_data(username):
#     """Loads user data from a JSON file."""
#     filepath = os.path.join(USER_DATA_DIR, f"{username}.json")
#     if os.path.exists(filepath):
#         try:
#             with open(filepath, 'r') as f:
#                 return json.load(f)
#         except json.JSONDecodeError:
#             logger.error(f"Error decoding JSON for user {username}")
#             return None # Or initialize with default structure
#     return None # Or initialize with default structure like {"username": username, "history": [], "performance_summary": {}}


# def save_user_data(username, data):
#     """Saves user data to a JSON file."""
#     filepath = os.path.join(USER_DATA_DIR, f"{username}.json")
#     try:
#         with open(filepath, 'w') as f:
#             json.dump(data, f, indent=4)
#     except Exception as e:
#         logger.error(f"Error saving data for user {username}: {e}")


# def initialize_user_data(username):
#     """Initializes data structure for a new user."""
#     return {
#         "username": username,
#         "quiz_history": [], # List of quiz attempt details
#         "topic_performance": {} # Key: Main Topic, Value: { "attempts": N, "avg_score": X, "current_difficulty": "medium", "subtopic_mastery": {"subtopic1": 0.8, "subtopic2": 0.3}}
#     }

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
geminimodel = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
model = SentenceTransformer("all-MiniLM-L6-v2")


app = Flask(__name__)
app.secret_key = 'dev_key_123' # secret key for session management in chat functionality, should be changed in production


# UPLOAD_FOLDER and ALLOWED_EXTENSIONS seem unused in the provided code for book processing
# They might be for a different feature (e.g., user PDF uploads)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
# --- SQLAlchemy Configuration ---
# Replace with your PostgreSQL connection string
# Format: postgresql://username:password@host:port/database_name
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://your_user:your_password@localhost:5432/quiz_app_db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Suppress a warning

    
# --- Now initialize db with app and then Migrate ---
db.init_app(app)
migrate = Migrate(app, db) # Initialize Flask-Migrate


# --- Database Helper Functions (Modified/New) ---
def get_or_create_user(username_str):
    user = User.query.filter_by(username=username_str).first()
    if not user:
        user = User(username=username_str)
        db.session.add(user)
        try:
            db.session.commit()
            logger.info(f"Created new user: {username_str} with ID: {user.id}")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating user {username_str}: {e}")
            raise # Re-raise the exception to be handled by the route
    return user

def get_or_create_topic_performance(user_id, topic_name_str):
    topic_perf = TopicPerformance.query.filter_by(user_id=user_id, topic_name=topic_name_str).first()
    if not topic_perf:
        topic_perf = TopicPerformance(user_id=user_id, topic_name=topic_name_str)
        db.session.add(topic_perf)
        # Commit might be better done in a batch with other changes,
        # but for get_or_create, committing here ensures it exists for subsequent reads.
        try:
            db.session.commit()
            logger.info(f"Created new TopicPerformance for user_id {user_id} and topic '{topic_name_str}'")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating TopicPerformance for {user_id}, {topic_name_str}: {e}")
            raise
    return topic_perf

def get_or_create_subtopic_mastery(topic_performance_id, subtopic_name_str):
    sub_mastery = SubtopicMastery.query.filter_by(
        topic_performance_id=topic_performance_id,
        subtopic_name=subtopic_name_str
    ).first()
    if not sub_mastery:
        sub_mastery = SubtopicMastery(
            topic_performance_id=topic_performance_id,
            subtopic_name=subtopic_name_str,
            times_seen=0,  # Explicitly set to 0
            correct_answers=0  # Explicitly set to 0
        )
        db.session.add(sub_mastery)
        # Usually committed along with other updates in the submit route
    return sub_mastery


# Define base directories for organized files
PDF_DIR = "pdf_files"
INDEX_DIR = "index_files"
# Ensure these directories exist if needed for writing, though reading is the primary focus here
# os.makedirs(PDF_DIR, exist_ok=True) # Will be created in if __name__ == '__main__'
# os.makedirs(INDEX_DIR, exist_ok=True) # Will be created in if __name__ == '__main__'


# --- Pre-compile regexes for generate_questions ---
RE_QUESTION_STEM = re.compile(r'Q\d+:\s*(.*?)(?=\n\s*[aA]\))', re.DOTALL | re.IGNORECASE)
RE_OPTIONS_BLOCK = re.compile(
    r'[aA]\)\s*(?P<option_a>.*?)\s*'
    r'[bB]\)\s*(?P<option_b>.*?)\s*'
    r'[cC]\)\s*(?P<option_c>.*?)\s*'
    r'[dD]\)\s*(?P<option_d>.*?)\s*'
    r'(?=\n\s*(?:Q\d+|Answer List:)|\Z)',
    re.DOTALL | re.IGNORECASE
)
RE_ANSWER_LIST_STR = re.compile(r"Answer List:\s*(\[.*?\])", re.DOTALL | re.IGNORECASE)
# --- End of pre-compiled regexes for generate_questions ---





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

# Modify the existing generate_questions function
def generate_questions(topic, num_questions=10, difficulty="medium"): # Added num_questions and difficulty
    # Updated prompt for clarity, difficulty, and to guide LLM on answer format
    prompt = f"""You are an expert in {topic}. Your task is to generate {num_questions} multiple-choice questions (MCQs) of {difficulty} difficulty, covering diverse topics within this subject.
    Each question must be followed by four answer choices (a, b, c, d), and the correct answer must be among them.
    Ensure each question stem is clearly identifiable (e.g., ending with a colon or question mark before the options).

    Output Format:
    Q1: [Question Text]?
    a) [Option 1]
    b) [Option 2]
    c) [Option 3]
    d) [Option 4]

    ... (up to Q{num_questions})

    Answer List: ['Correct letter for Q1', 'Correct letter for Q2', ..., 'Correct letter for Q{num_questions}'] (e.g., ['a', 'c', 'd', ...])
    Alternatively, the Answer List can be: ['Letter) Full text for Q1', 'Letter) Full text for Q2', ...]
    """
    try:
        response = geminimodel.generate_content(prompt)
        if not response.parts:
            logger.error("No response parts received from Gemini model.")
            return [], 0
        text = response.text
    except Exception as e:
        logger.error(f"Error generating content from Gemini: {e}")
        return [], 0

    logger.debug("-----------------RAW LLM OUTPUT--------------------")
    logger.debug(text)
    logger.debug("-------------------------------------------------")

    # ... (rest of the parsing logic remains largely the same)
    # Ensure you parse up to num_questions

    # The core parsing logic:
    questions = [q.strip() for q in RE_QUESTION_STEM.findall(text)]
    parsed_options_list = []
    for match in RE_OPTIONS_BLOCK.finditer(text):
        opts = [
            match.group('option_a').strip(),
            match.group('option_b').strip(),
            match.group('option_c').strip(),
            match.group('option_d').strip()
        ]
        parsed_options_list.append(opts)

    answer_list_match = RE_ANSWER_LIST_STR.search(text)
    llm_correct_answers_raw = []
    if answer_list_match:
        try:
            raw_list_str = answer_list_match.group(1)
            llm_correct_answers_raw = ast.literal_eval(raw_list_str)
            llm_correct_answers_raw = [str(ans).strip() for ans in llm_correct_answers_raw]
        except (ValueError, SyntaxError) as e:
            logger.error(f"Error parsing Answer List string: {e}. Raw string: '{raw_list_str if 'raw_list_str' in locals() else 'not found'}'")
            llm_correct_answers_raw = []
    else:
        logger.warning("Could not find 'Answer List:' in the LLM output.")

    num_questions_found = len(questions)
    num_options_sets_found = len(parsed_options_list)
    num_answers_found = len(llm_correct_answers_raw)

    logger.debug(f"Found: {num_questions_found} questions, {num_options_sets_found} option sets, {num_answers_found} raw answers (requested: {num_questions}).")

    # Align based on the *requested* num_questions and what was parsed
    # We want exactly num_questions if possible, or the minimum of parsed items up to num_questions
    max_possible = min(num_questions_found, num_options_sets_found, num_answers_found)
    final_count = min(max_possible, num_questions)


    if final_count < num_questions and max_possible > 0 : # LLM didn't provide enough, or parsing issue
        logger.warning(
            f"Requested {num_questions} but could only finalize {final_count} "
            f"(Parsed Q:{num_questions_found}, O:{num_options_sets_found}, A:{num_answers_found}). Truncating."
        )
    elif final_count == 0:
         logger.error("No valid question data could be parsed to meet request.")
         return [], 0

    questions = questions[:final_count]
    parsed_options_list = parsed_options_list[:final_count]
    llm_correct_answers_raw = llm_correct_answers_raw[:final_count]

    logger.info(f"Successfully parsed {final_count} question structures.")
    question_data = []
    letter_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

    for i in range(final_count):
        q_text = questions[i]
        current_options = parsed_options_list[i]
        llm_ans_item = llm_correct_answers_raw[i].lower()
        determined_correct_text = ""

        # ... (Your existing robust answer matching logic) ...
        if len(llm_ans_item) == 1 and llm_ans_item in letter_to_index:
            correct_idx = letter_to_index[llm_ans_item]
            if 0 <= correct_idx < len(current_options):
                determined_correct_text = current_options[correct_idx]
                # logger.debug(f"Q{i+1} ans '{llm_ans_item}': Matched as letter, maps to option {correct_idx+1}: '{determined_correct_text[:50]}...'")

        if not determined_correct_text:
            for letter_key, correct_idx in letter_to_index.items():
                if llm_ans_item.startswith(letter_key + ")") or \
                   llm_ans_item.startswith(letter_key + "."):
                    if 0 <= correct_idx < len(current_options):
                        determined_correct_text = current_options[correct_idx]
                        # logger.debug(f"Q{i+1} ans '{llm_ans_item[:50]}...': Matched as letter-prefix, maps to option {correct_idx+1}: '{determined_correct_text[:50]}...'")
                        break

        if not determined_correct_text:
            potential_text_from_llm = llm_ans_item
            for letter_key in letter_to_index.keys():
                if potential_text_from_llm.startswith(letter_key + ")"):
                    potential_text_from_llm = potential_text_from_llm[len(letter_key)+1:].strip()
                    break
                if potential_text_from_llm.startswith(letter_key + "."):
                    potential_text_from_llm = potential_text_from_llm[len(letter_key)+1:].strip()
                    break
            for opt_idx, opt_text_candidate in enumerate(current_options):
                # Case-insensitive comparison for full text match
                if potential_text_from_llm.lower() == opt_text_candidate.lower():
                    determined_correct_text = opt_text_candidate
                    # logger.debug(f"Q{i+1} ans '{llm_ans_item[:50]}...': Matched as full text (prefix stripped: '{potential_text_from_llm[:50]}...'), maps to option {opt_idx+1}: '{determined_correct_text[:50]}...'")
                    break

        if not determined_correct_text:
            logger.warning(
                f"Q{i+1}: Could not definitively match LLM answer '{llm_correct_answers_raw[i]}' "
                f"to options {current_options}. Defaulting to the first option."
            )
            determined_correct_text = current_options[0] if current_options else ""

        if current_options and determined_correct_text not in current_options:
            # Final check, if determined_correct_text somehow isn't in options (should be rare with good parsing)
            is_match_found_case_insensitive = False
            for opt in current_options:
                if opt.lower() == determined_correct_text.lower():
                    determined_correct_text = opt # Use the casing from options
                    is_match_found_case_insensitive = True
                    break
            if not is_match_found_case_insensitive:
                logger.error(
                    f"Q{i+1} FATAL PARSE ERROR: Determined correct answer '{determined_correct_text}' "
                    f"is NOT in the parsed options {current_options} (even case-insensitively). This indicates a flaw in parsing logic. "
                    f"LLM raw answer was: '{llm_correct_answers_raw[i]}'. Defaulting to first option."
                )
                determined_correct_text = current_options[0] if current_options else ""


        question_data.append({
            "question": q_text,
            "options": current_options,
            "correct": determined_correct_text
        })

    logger.info(f"Prepared {len(question_data)} questions for the quiz.")
    return question_data, len(question_data)


def get_subtopic(question):
    # prompt = f"Analyze the following question and classify it into a single, concise subtopic or medical specialty. Examples: Cardiology, Infectious Disease, Pharmacology, Renal Physiology, etc.\n\nQuestion: {question}\n\nSubtopic:"
    
    prompt = f"Consider the following question. Determine the primary subject area or the most specific, yet broadly understandable, subtopic it directly addresses. Provide a single, concise label for this classification. Question{question} Classification:"

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

        # Using the same improved parsing logic as generate_questions
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
        5. Ensure each question stem ends with a colon or question mark before the options.

        Output Format:
        Q1: [Question 1]?
        a) [Option 1a]
        b) [Option 1b]
        c) [Option 1c]
        d) [Option 1d]

        ... (up to Q5)

        Answer List: ['Correct letter for Q1', 'Correct letter for Q2', ..., 'Correct letter for Q5'] (e.g., ['a', 'c', 'd', 'b', 'a'])
        Alternatively, the Answer List can be: ['Letter) Full text for Q1', 'Letter) Full text for Q2', ...]
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

        # logger.debug("-----------------BOOK QUESTIONS RAW LLM OUTPUT--------------------")
        # logger.debug(text) # Uncomment for debugging
        # logger.debug("----------------------------------------------------------------")

        # Re-using the pre-compiled regexes for parsing book questions
        questions = [q.strip() for q in RE_QUESTION_STEM.findall(text)][:5] # Limit to 5
        parsed_options_list = []
        for match in RE_OPTIONS_BLOCK.finditer(text):
            if len(parsed_options_list) < 5: # Only take up to 5 sets of options
                opts = [
                    match.group('option_a').strip(),
                    match.group('option_b').strip(),
                    match.group('option_c').strip(),
                    match.group('option_d').strip()
                ]
                parsed_options_list.append(opts)
        
        answer_list_match = RE_ANSWER_LIST_STR.search(text)
        llm_correct_answers_raw = []
        if answer_list_match:
            try:
                raw_list_str = answer_list_match.group(1)
                llm_correct_answers_raw = ast.literal_eval(raw_list_str)
                llm_correct_answers_raw = [str(ans).strip() for ans in llm_correct_answers_raw][:5] # Limit to 5
            except (ValueError, SyntaxError) as e:
                logger.error(f"Error parsing Answer List for book questions: {e}. Raw: '{raw_list_str if 'raw_list_str' in locals() else 'not found'}'")
                llm_correct_answers_raw = []
        else:
            logger.warning("Could not find 'Answer List:' in the book questions LLM output.")


        num_q = len(questions)
        num_o = len(parsed_options_list)
        num_a = len(llm_correct_answers_raw)

        logger.debug(f"Book questions parsed: Q:{num_q}, O_sets:{num_o}, A_raw:{num_a}")

        # Align counts for book questions (max 5)
        final_book_q_count = 0
        if not (num_q == num_o == num_a and num_q > 0):
            logger.warning(f"Book questions: Mismatch in parsed items. Q:{num_q}, O:{num_o}, A:{num_a}. Attempting to align.")
            final_book_q_count = min(num_q, num_o, num_a)
            # Further cap at 5 if min_count is higher due to regex over-matching before slicing
            final_book_q_count = min(final_book_q_count, 5)
            if final_book_q_count < 5:
                 logger.warning(f"Book question data will be truncated to {final_book_q_count} items.")
            questions = questions[:final_book_q_count]
            parsed_options_list = parsed_options_list[:final_book_q_count]
            llm_correct_answers_raw = llm_correct_answers_raw[:final_book_q_count]
        elif num_q > 0:
            final_book_q_count = min(num_q, 5) # Ensure we don't exceed 5 even if all match
        else:
            logger.error("No valid book question data could be parsed.")
            return []


        question_data = []
        letter_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3} # Re-define locally for clarity

        for i in range(final_book_q_count):
            q_text = questions[i]
            current_options = parsed_options_list[i]
            llm_ans_item = llm_correct_answers_raw[i].lower()
            determined_correct_text = ""

            if len(llm_ans_item) == 1 and llm_ans_item in letter_to_index:
                correct_idx = letter_to_index[llm_ans_item]
                if 0 <= correct_idx < len(current_options):
                    determined_correct_text = current_options[correct_idx]

            if not determined_correct_text:
                for letter_key, correct_idx in letter_to_index.items():
                    if llm_ans_item.startswith(letter_key + ")") or \
                       llm_ans_item.startswith(letter_key + "."):
                        if 0 <= correct_idx < len(current_options):
                            determined_correct_text = current_options[correct_idx]
                            break
            
            if not determined_correct_text:
                potential_text_from_llm = llm_ans_item
                for letter_key in letter_to_index.keys():
                    if potential_text_from_llm.startswith(letter_key + ")"):
                        potential_text_from_llm = potential_text_from_llm[len(letter_key)+1:].strip()
                        break
                    if potential_text_from_llm.startswith(letter_key + "."):
                        potential_text_from_llm = potential_text_from_llm[len(letter_key)+1:].strip()
                        break
                for opt_text_candidate in current_options:
                    if potential_text_from_llm == opt_text_candidate.lower():
                        determined_correct_text = opt_text_candidate
                        break
            
            if not determined_correct_text:
                logger.warning(f"Book Q{i+1}: Could not match LLM answer '{llm_correct_answers_raw[i]}' to options. Defaulting.")
                determined_correct_text = current_options[0] if current_options else ""

            if current_options and determined_correct_text not in current_options:
                 logger.error(f"Book Q{i+1} FATAL PARSE: Correct answer '{determined_correct_text}' not in options {current_options}. LLM raw: '{llm_correct_answers_raw[i]}'. Defaulting.")
                 determined_correct_text = current_options[0] if current_options else ""

            question_data.append({
                "question": q_text,
                "options": current_options,
                "correct": determined_correct_text
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


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         username = request.form.get('username', 'guest').strip().lower()
#         if not username: username = 'guest' # Default username
#         session['username'] = username # Store username in session

#         topic = request.form.get('topic', 'General Knowledge').strip()
#         if not topic: topic = 'General Knowledge'
        
#         user_data = load_user_data(username)
#         if not user_data:
#             user_data = initialize_user_data(username)
#             save_user_data(username, user_data) # Save initial structure

#         # Determine difficulty for this topic based on user's history
#         difficulty = "medium" # Default
#         num_questions_main_quiz = 10 # Default for main quiz

#         if topic in user_data.get("topic_performance", {}):
#             difficulty = user_data["topic_performance"][topic].get("current_difficulty", "medium")
#             # Potentially adjust num_questions too, or keep it fixed for main topics

#         logger.info(f"User: {username}, Generating quiz for topic: {topic}, Difficulty: {difficulty}")
        
#         # Store current quiz context in session for the submit route
#         session['current_quiz_topic'] = topic
#         session['current_quiz_difficulty'] = difficulty
#         session['current_quiz_type'] = "main" # To distinguish from remedial

#         # questions, count = generate_questions(topic, num_questions=num_questions_main_quiz, difficulty=difficulty)
        
#         questions_python_list, count = generate_questions(topic, num_questions=num_questions_main_quiz, difficulty=difficulty) # Renamed for clarity
        
#         if not questions_python_list:
#             return render_template('index.html', error="Could not generate questions. Please try again.")
        
#         # Explicitly create the JSON string for the hidden field
#         # Ensure json is imported at the top of your file: import json
#         questions_json_string = json.dumps(questions_python_list)
        
        
#         # return render_template('quiz.html', 
#         #                        questions=questions_python_list, 
#         #                        topic=topic, count=count, 
#         #                        username=username, 
#         #                        difficulty=difficulty)
        
#         return render_template('quiz.html', 
#                                questions_data_for_template=questions_python_list,  # This is the Python list for display
#                                questions_json_for_hidden_field=questions_json_string, # This is the JSON string for the form
#                                topic=topic, 
#                                count=count, 
#                                username=username, 
#                                difficulty=difficulty)
    
#     # For GET request, if a user is already in session, prefill username
#     username_in_session = session.get('username', '')
#     return render_template('index.html', username_in_session=username_in_session)




# @app.route('/submit', methods=['POST'])
# def submit():
#     logger.info("Processing quiz submission.")
#     username = session.get('username')
#     if not username:
#         logger.error("Username not found in session during submit.")
#         return redirect(url_for('index')) # Or an error page

#     current_topic = session.get('current_quiz_topic', 'Unknown Topic')
#     current_difficulty = session.get('current_quiz_difficulty', 'medium')
#     # current_quiz_type = session.get('current_quiz_type', 'main') # Useful if remedial quizzes update different stats

#     user_data = load_user_data(username)
#     if not user_data: # Should not happen if index initialized it
#         user_data = initialize_user_data(username)

#     try:
#         questions_str = request.form.get('questions')
#         logger.debug(f"Received questions_str: {questions_str}") # <--- ADD THIS LINE
#         if not questions_str:
#              logger.error("No questions data received in submission.")
#              return "Error: Missing question data in submission.", 400
#         # original_questions_data = ast.literal_eval(questions_str) # These are the Qs sent to user
#         original_questions_data = json.loads(questions_str)


#         if not isinstance(original_questions_data, list):
#              logger.error("Invalid format for questions data received.")
#              return "Error: Invalid question data format.", 400
#     except Exception as e:
#          logger.error(f"Error processing form questions data: {e}", exc_info=True)
#          return "Server error during submission processing.", 500

#     score = 0
#     total = len(original_questions_data)
#     quiz_attempt_details = {
#         "quiz_id": str(uuid.uuid4()),
#         "topic": current_topic,
#         "difficulty": current_difficulty,
#         "timestamp": datetime.utcnow().isoformat(),
#         "score": 0,
#         "total_questions": total,
#         "questions_answered": [] # Detailed per-question info
#     }
    
#     weak_subtopics_this_quiz = set() # Use a set to store unique subtopics
#     weak_questions_details_for_explanation = [] # For the explanation generation

#     for i, q_data in enumerate(original_questions_data):
#         selected_answer = request.form.get(f'question{i}', '').strip()
#         correct_answer = q_data.get('correct', '').strip()
#         question_text = q_data.get('question', 'Unknown Question')
#         options = q_data.get('options', [])
        
#         is_correct = (selected_answer == correct_answer)
#         if is_correct:
#             score += 1
#         else:
#             subtopic = get_subtopic(question_text) # This is your existing function
#             if subtopic not in ["Unknown Subtopic", "Subtopic Error"]: # Filter out unhelpfuls
#                  weak_subtopics_this_quiz.add(subtopic)
#             weak_questions_details_for_explanation.append({
#                 "question": question_text,
#                 "options": options,
#                 "selected": selected_answer,
#                 "correct": correct_answer
#             })

#         quiz_attempt_details["questions_answered"].append({
#             "question_text": question_text,
#             "options": options,
#             "selected_answer": selected_answer,
#             "correct_answer": correct_answer,
#             "is_correct": is_correct,
#             "subtopic": get_subtopic(question_text) if not is_correct else None # Only get subtopic if wrong
#         })

#     quiz_attempt_details["score"] = score
#     user_data["quiz_history"].append(quiz_attempt_details)

#     # --- Update overall topic performance ---
#     if current_topic not in user_data["topic_performance"]:
#         user_data["topic_performance"][current_topic] = {
#             "attempts": 0,
#             "total_score_points": 0, # To calculate average later
#             "total_questions_attempted": 0, # To calculate average later
#             "current_difficulty": "medium",
#             "subtopic_mastery": {} # Subtopic: times_correct / times_seen
#         }
    
#     topic_perf = user_data["topic_performance"][current_topic]
#     topic_perf["attempts"] += 1
#     topic_perf["total_score_points"] += score
#     topic_perf["total_questions_attempted"] += total
    
#     # Update subtopic mastery
#     for q_answered in quiz_attempt_details["questions_answered"]:
#         sub = q_answered.get("subtopic") # This will be None if correct, or the determined subtopic if wrong
#                                         # For mastery, we need subtopics for ALL questions. Let's re-fetch if needed.
#         if not sub and not q_answered["is_correct"]: # Should have been fetched if wrong
#             sub = get_subtopic(q_answered["question_text"])
#         elif q_answered["is_correct"]: # If correct, we still need its subtopic for mastery calculation
#             sub = get_subtopic(q_answered["question_text"])

#         if sub and sub not in ["Unknown Subtopic", "Subtopic Error"]:
#             if sub not in topic_perf["subtopic_mastery"]:
#                 topic_perf["subtopic_mastery"][sub] = {"correct": 0, "seen": 0}
#             topic_perf["subtopic_mastery"][sub]["seen"] += 1
#             if q_answered["is_correct"]:
#                 topic_perf["subtopic_mastery"][sub]["correct"] += 1

#     # --- Adjust difficulty for next time on this MAIN topic ---
#     # This is a simple rule, can be more sophisticated
#     if total > 0: # Avoid division by zero
#         current_quiz_score_percent = (score / total) * 100
#         if current_quiz_score_percent >= 80 and current_difficulty == "easy":
#             topic_perf["current_difficulty"] = "medium"
#         elif current_quiz_score_percent >= 70 and current_difficulty == "medium":
#             topic_perf["current_difficulty"] = "hard"
#         elif current_quiz_score_percent < 40 and current_difficulty == "hard":
#             topic_perf["current_difficulty"] = "medium"
#         elif current_quiz_score_percent < 30 and current_difficulty == "medium":
#             topic_perf["current_difficulty"] = "easy"
#         # else, keep current difficulty

#     save_user_data(username, user_data)

#     # --- Generate Explanations (Your existing logic) ---
#     explanation_list = []
#     # ... (your explanation generation loop using weak_questions_details_for_explanation) ...
#     logger.info(f"Generating explanations for {len(weak_questions_details_for_explanation)} incorrect answers.")
#     for detail in weak_questions_details_for_explanation:
#         prompt = f"""The user incorrectly answered the following multiple-choice question:
#         Question: {detail['question']} Options: ... User's incorrect answer: {detail['selected']} Correct answer: {detail['correct']}
#         Please provide a concise and clear explanation...""" # (Your full prompt)
#         try:
#             response = geminimodel.generate_content(prompt)
#             explanation = response.text.strip() if response.parts else "Could not generate explanation."
#             explanation_list.append(explanation)
#         except Exception as e:
#             logger.error(f"Error generating explanation: {e}")
#             explanation_list.append("Error generating explanation.")

#     question_explanations = list(zip([d['question'] for d in weak_questions_details_for_explanation], explanation_list))
    
#     # SERP API articles (Your existing logic)
#     # --- Optional: Fetch articles using SERP API for weak topics ---
#     # Be mindful of API costs and rate limits
#     articles_list = []
#     USE_SERP_API = False # Set to True to enable SERP API calls
#     if USE_SERP_API and weak_subtopics_this_quiz:
#          logger.info("Fetching related articles using SERP API...")
#          # Use unique weak topics to avoid redundant searches
#          unique_weak_topics = list(set(weak_subtopics_this_quiz) - {"Unknown Subtopic", "Subtopic Error"}) # Exclude generic/error topics
#          for weak_topic in unique_weak_topics[:3]: # Limit number of searches
#              logger.debug(f"Searching SERP API for: {weak_topic}")
#              try:
#                  params = {
#                      "q": f"{weak_topic} learning resources OR explanation", # More specific query
#                      "hl": "en",
#                      "api_key": os.getenv("SERP_API_KEY")
#                  }
#                  if not params["api_key"]:
#                       logger.warning("SERP_API_KEY not found in environment variables. Skipping article search.")
#                       break # Stop trying if key is missing

#                  search = GoogleSearch(params)
#                  results = search.get_dict()

#                  if "organic_results" in results:
#                      for result in results["organic_results"][:2]: # Get top 2 links per topic
#                          articles_list.append({
#                              "topic": weak_topic, # Changed from 'question' to 'topic'
#                              "title": result.get('title', 'No Title'),
#                              "link": result.get('link', '#'),
#                              "snippet": result.get('snippet', '')
#                          })
#                  # Optional delay between API calls
#                  # time.sleep(1)
#              except Exception as e:
#                  logger.error(f"Error fetching SERP API results for '{weak_topic}': {e}")



#     logger.info(f"Quiz for {username} on {current_topic} finished. Score: {score}/{total}. Weak subtopics: {weak_subtopics_this_quiz}")
    
#     return render_template('result.html',
#                            score=score,
#                            total=total,
#                            username=username,
#                            topic=current_topic,
#                            weak_subtopics_identified=list(weak_subtopics_this_quiz), # Pass to template
#                            articles=articles_list,
#                            question_explanations=question_explanations,
#                            next_difficulty_for_topic=topic_perf["current_difficulty"])




# @app.route('/remedial_quiz', methods=['GET', 'POST'])
# def remedial_quiz():
#     username = session.get('username')
#     if not username:
#         return redirect(url_for('index'))

#     if request.method == 'POST': # This means the user clicked "Start Remedial Quiz"
#         sub_topic_to_remediate = request.form.get('sub_topic')
#         main_topic_context = request.form.get('main_topic') # The original topic for context
        
#         if not sub_topic_to_remediate or not main_topic_context:
#             # Handle error, maybe redirect back to results or index
#             return redirect(url_for('index'))

#         # Remedial quizzes could be a fixed number of questions and medium difficulty, or adapt too
#         remedial_difficulty = "medium" # Or base it on subtopic_mastery
#         num_remedial_questions = 5 # Fewer questions for focused topic

#         logger.info(f"User: {username}, Generating REMEDIAL quiz for sub-topic: {sub_topic_to_remediate} (context: {main_topic_context}), Difficulty: {remedial_difficulty}")
        
#         # Store current quiz context for the submit route
#         # Important: Distinguish this from a "main" quiz submission if you want separate handling
#         session['current_quiz_topic'] = f"{main_topic_context} (Remedial: {sub_topic_to_remediate})"
#         session['current_quiz_difficulty'] = remedial_difficulty
#         session['current_quiz_type'] = "remedial" 
#         session['current_remedial_subtopic'] = sub_topic_to_remediate # For specific updates if needed

#         # The generate_questions function needs to understand combined topic/subtopic
#         # Or, ensure the LLM understands the subtopic focus from a more general topic prompt
#         # For now, let's pass the sub_topic_to_remediate as the main topic for the LLM
#         questions, count = generate_questions(
#             topic=f"{sub_topic_to_remediate} (related to {main_topic_context})", # More context for LLM
#             num_questions=num_remedial_questions, 
#             difficulty=remedial_difficulty
#         )
        
#         if not questions:
#             return render_template('index.html', error="Could not generate questions. Please try again.")
        
#         # Explicitly create the JSON string here
#         questions_json_string = json.dumps(questions)
        
#         return render_template('quiz.html',
#                        questions_data_for_template=questions, # For iterating and displaying
#                        questions_json_for_hidden_field=questions_json_string, # For the hidden field
#                        topic=f"Remedial: {sub_topic_to_remediate}", # Display topic
#                        count=count,
#                        username=username,
#                        difficulty=remedial_difficulty)
    
#     # If GET, it means this route was likely accessed incorrectly without POST data
#     return redirect(url_for('index'))







# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username_str = request.form.get('username', 'guest').strip().lower()
        if not username_str: username_str = 'guest'
        session['username'] = username_str # Keep in session for easy access

        topic_name = request.form.get('topic', 'General Knowledge').strip()
        if not topic_name: topic_name = 'General Knowledge'
        
        try:
            user = get_or_create_user(username_str) # Get User DB object
            # Get or create TopicPerformance for this user and topic
            topic_perf = get_or_create_topic_performance(user.id, topic_name)
        except Exception as e:
            logger.error(f"Database error in index for user {username_str}, topic {topic_name}: {e}")
            return render_template('index.html', error="A database error occurred. Please try again later.")


        difficulty = topic_perf.current_difficulty # Default is 'medium' from model
        num_questions_main_quiz = 10

        logger.info(f"User: {user.username} (ID: {user.id}), Generating quiz for topic: {topic_name}, Difficulty: {difficulty}")
        
        session['current_quiz_topic'] = topic_name
        session['current_quiz_difficulty'] = difficulty
        session['current_quiz_type'] = "main"
        session['user_db_id'] = user.id # Store user's DB ID in session

        questions_python_list, count = generate_questions(topic_name, num_questions=num_questions_main_quiz, difficulty=difficulty)
        
        if not questions_python_list:
            return render_template('index.html', error="Could not generate questions. Please try again.")
        
        questions_json_string = json.dumps(questions_python_list)
        
        return render_template('quiz.html', 
                               questions_data_for_template=questions_python_list,
                               questions_json_for_hidden_field=questions_json_string,
                               topic=topic_name, 
                               count=count, 
                               username=user.username, 
                               difficulty=difficulty)
    
    username_in_session = session.get('username', '')
    return render_template('index.html', username_in_session=username_in_session)

@app.route('/submit', methods=['POST'])
def submit():
    logger.info("Processing quiz submission.")
    
    # Retrieve from session (set in index or remedial_quiz)
    username_str = session.get('username')
    user_db_id = session.get('user_db_id') # Get user's DB ID
    current_topic_name = session.get('current_quiz_topic', 'Unknown Topic')
    current_difficulty = session.get('current_quiz_difficulty', 'medium')

    if not username_str or not user_db_id:
        logger.error("Username or User DB ID not found in session during submit.")
        return redirect(url_for('index'))

    try:
        questions_str = request.form.get('questions')
        # logger.debug(f"Received questions_str from form: {questions_str}")
        if not questions_str:
             logger.error("No questions data received in submission.")
             return "Error: Missing question data in submission.", 400
        original_questions_data = json.loads(questions_str)

        if not isinstance(original_questions_data, list):
             logger.error("Invalid format for questions data received.")
             return "Error: Invalid question data format.", 400
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON questions_str: {e}. String was: {questions_str}")
        return "Error processing submission data (JSON decode failed).", 400
    except Exception as e:
         logger.error(f"Error processing form questions data: {e}", exc_info=True)
         return "Server error during submission processing.", 500

    score = 0
    total_questions_in_quiz = len(original_questions_data)
    
    # Create QuizAttempt record
    new_quiz_attempt = QuizAttempt(
        user_id=user_db_id,
        topic_name=current_topic_name,
        difficulty=current_difficulty,
        score=0, # Will be updated
        total_questions=total_questions_in_quiz
    )
    db.session.add(new_quiz_attempt)
    # We need to flush to get new_quiz_attempt.id for AnsweredQuestion,
    # but commit will be done at the end.
    try:
        db.session.flush() # Assigns ID to new_quiz_attempt without full commit
    except Exception as e:
        db.session.rollback()
        logger.error(f"DB error flushing new quiz attempt: {e}")
        return "Database error during submission. Please try again."


    weak_subtopics_this_quiz = set()
    weak_questions_details_for_explanation = []
    
    answered_questions_to_add = [] # For batch adding later

    for i, q_data_from_form in enumerate(original_questions_data):
        selected_answer = request.form.get(f'question{i}', '').strip()
        correct_answer = q_data_from_form.get('correct', '').strip()
        question_text = q_data_from_form.get('question', 'Unknown Question')
        options_list = q_data_from_form.get('options', []) # This is a list

        # Fetch subtopic for this question (call once per question)
        question_subtopic = get_subtopic(question_text)

        is_correct = (selected_answer == correct_answer)
        if is_correct:
            score += 1
        else:
            if question_subtopic not in ["Unknown Subtopic", "Subtopic Error"]:
                 weak_subtopics_this_quiz.add(question_subtopic)
            weak_questions_details_for_explanation.append({
                "question": question_text,
                "options": options_list,
                "selected": selected_answer,
                "correct": correct_answer
            })
        
        ans_q = AnsweredQuestion(
            quiz_attempt_id=new_quiz_attempt.id,
            question_text=question_text,
            options=options_list, # Store as JSON (SQLAlchemy handles Python list to JSON)
            selected_answer=selected_answer,
            correct_answer=correct_answer,
            is_correct=is_correct,
            subtopic=question_subtopic if question_subtopic not in ["Unknown Subtopic", "Subtopic Error"] else None
        )
        answered_questions_to_add.append(ans_q)

    new_quiz_attempt.score = score # Update score
    db.session.add_all(answered_questions_to_add) # Add all answered questions

    # --- Update TopicPerformance and SubtopicMastery ---
    try:
        topic_perf = get_or_create_topic_performance(user_db_id, current_topic_name)
        topic_perf.attempts += 1
        topic_perf.total_score_points += score
        topic_perf.total_questions_attempted += total_questions_in_quiz
        
        # Update subtopic mastery for this topic_perf
        for aq_obj in answered_questions_to_add: # Iterate over the AnsweredQuestion objects we just created
            if aq_obj.subtopic: # If a valid subtopic was determined
                sub_mastery = get_or_create_subtopic_mastery(topic_perf.id, aq_obj.subtopic)
                sub_mastery.times_seen += 1
                if aq_obj.is_correct:
                    sub_mastery.correct_answers += 1
                db.session.add(sub_mastery) # Add to session if new or to update if existing

        # Adjust difficulty for next time on this topic
        if total_questions_in_quiz > 0:
            current_quiz_score_percent = (score / total_questions_in_quiz) * 100
            if current_quiz_score_percent >= 80 and topic_perf.current_difficulty == "easy":
                topic_perf.current_difficulty = "medium"
            elif current_quiz_score_percent >= 70 and topic_perf.current_difficulty == "medium":
                topic_perf.current_difficulty = "hard"
            elif current_quiz_score_percent < 40 and topic_perf.current_difficulty == "hard":
                topic_perf.current_difficulty = "medium"
            elif current_quiz_score_percent < 30 and topic_perf.current_difficulty == "medium":
                topic_perf.current_difficulty = "easy"
        
        db.session.add(topic_perf) # Add to session to update
        db.session.commit() # Commit all changes: QuizAttempt, AnsweredQuestions, TopicPerformance, SubtopicMastery
        logger.info(f"Successfully committed quiz attempt and performance for user {user_db_id}, topic {current_topic_name}")

    except Exception as e:
        db.session.rollback()
        logger.error(f"Database error during submit processing for user {user_db_id}: {e}", exc_info=True)
        return "A database error occurred while saving your results. Please try again.", 500

    # --- Generate Explanations (Your existing logic) ---
    explanation_list = []
    # ... (Your explanation generation code as before, using weak_questions_details_for_explanation) ...
    logger.info(f"Generating explanations for {len(weak_questions_details_for_explanation)} incorrect answers.")
    for detail in weak_questions_details_for_explanation:
        prompt = f"""The user incorrectly answered the following multiple-choice question:
        Question: {detail['question']} Options: a) {detail['options'][0] if len(detail['options']) > 0 else 'N/A'} b) {detail['options'][1] if len(detail['options']) > 1 else 'N/A'} c) {detail['options'][2] if len(detail['options']) > 2 else 'N/A'} d) {detail['options'][3] if len(detail['options']) > 3 else 'N/A'}
        User's incorrect answer: {detail['selected']} Correct answer: {detail['correct']}
        Please provide a concise and clear explanation...""" # Your full prompt
        try:
            response = geminimodel.generate_content(prompt)
            explanation = response.text.strip() if response.parts else "Could not generate explanation."
            explanation_list.append(explanation)
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            explanation_list.append("Error generating explanation.")
    question_explanations = list(zip([d['question'] for d in weak_questions_details_for_explanation], explanation_list))
    
    # --- SERP API (Your existing logic) ---
    articles_list = []
    USE_SERP_API = False # Set to True to enable SERP API calls
    # ... (Your SERP API code as before, using weak_subtopics_this_quiz) ...
    if USE_SERP_API and weak_subtopics_this_quiz:
         logger.info("Fetching related articles using SERP API...")
         # Use unique weak topics to avoid redundant searches
         unique_weak_topics = list(set(weak_subtopics_this_quiz) - {"Unknown Subtopic", "Subtopic Error"}) # Exclude generic/error topics
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


    logger.info(f"Quiz for {username_str} on {current_topic_name} finished. Score: {score}/{total_questions_in_quiz}. Weak subtopics: {weak_subtopics_this_quiz}")
    
    # Retrieve updated topic_perf for next_difficulty_for_topic to pass to template
    # This is already available as `topic_perf` if the try block succeeded.
    next_difficulty = topic_perf.current_difficulty if 'topic_perf' in locals() and topic_perf else "medium"

    return render_template('result.html',
                           score=score,
                           total=total_questions_in_quiz,
                           username=username_str,
                           topic=current_topic_name,
                           weak_subtopics_identified=list(weak_subtopics_this_quiz),
                           articles=articles_list,
                           question_explanations=question_explanations,
                           next_difficulty_for_topic=next_difficulty)


@app.route('/remedial_quiz', methods=['GET', 'POST'])
def remedial_quiz():
    username_str = session.get('username')
    user_db_id = session.get('user_db_id') # Get user's DB ID

    if not username_str or not user_db_id: # Check both
        logger.warning("User not fully identified for remedial quiz. Redirecting to start.")
        return redirect(url_for('index'))

    if request.method == 'POST':
        sub_topic_to_remediate = request.form.get('sub_topic')
        main_topic_context = request.form.get('main_topic')
        
        if not sub_topic_to_remediate or not main_topic_context:
            return redirect(url_for('index')) # Or show an error

        remedial_difficulty = "medium" # Default for remedial
        num_remedial_questions = 5
        
        # The topic name for remedial quizzes will be stored in the DB as this combined string
        remedial_topic_name = f"{main_topic_context} (Remedial: {sub_topic_to_remediate})"

        logger.info(f"User: {username_str} (ID: {user_db_id}), Generating REMEDIAL quiz for topic: {remedial_topic_name}, Difficulty: {remedial_difficulty}")
        
        session['current_quiz_topic'] = remedial_topic_name
        session['current_quiz_difficulty'] = remedial_difficulty
        session['current_quiz_type'] = "remedial" 
        # session['user_db_id'] is already set from the main quiz flow

        # LLM prompt uses the more specific sub-topic for question generation
        questions_python_list, count = generate_questions(
            topic=f"{sub_topic_to_remediate} (related to {main_topic_context})",
            num_questions=num_remedial_questions, 
            difficulty=remedial_difficulty
        )
        
        if not questions_python_list:
            # Handle error: could not generate remedial questions
            # You might want to redirect to the results page with a flash message
            return render_template('index.html', error=f"Could not generate remedial quiz for {sub_topic_to_remediate}. Please try again.")
        
        questions_json_string = json.dumps(questions_python_list)
            
        return render_template('quiz.html', 
                               questions_data_for_template=questions_python_list,
                               questions_json_for_hidden_field=questions_json_string,
                               topic=remedial_topic_name, # Display the full remedial topic name
                               count=count, 
                               username=username_str, 
                               difficulty=remedial_difficulty)
    
    return redirect(url_for('index'))




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
             # For `start_chat()`, the history is managed by the chat object.
             # We send just the new user input.
             logger.debug(f"Sending message to Gemini for chat {chat_id}: {user_input}")
             try:
                 response = chat_obj.send_message(user_input) # Send only the new message
                 bot_reply = response.text.strip()
                 logger.debug(f"Received reply from Gemini for chat {chat_id}: {bot_reply[:100]}...")

                 display_history.append(("You", user_input))
                 display_history.append(("Gemini", bot_reply))
                 session["display_history"] = display_history

             except Exception as e:
                 logger.error(f"Error sending message or receiving reply in chat {chat_id}: {e}")
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
    # os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # os.makedirs(USER_DATA_DIR, exist_ok=True) # Add this
    
    with app.app_context(): # Important for creating tables if not using migrations initially
        db.create_all() # Creates tables based on your models if they don't exist
                        # For production, use Alembic migrations.
                        
    app.run(debug=True, use_reloader=False)