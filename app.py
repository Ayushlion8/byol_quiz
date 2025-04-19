import google.generativeai as genai
import re
import os
import ast
from itertools import zip_longest
import pandas as pd
from serpapi import GoogleSearch
import faiss
from flask import session,redirect
from sentence_transformers import SentenceTransformer
import numpy as np
import cv2
from flask import Flask, render_template, Response, request, jsonify,url_for
from PIL import Image
from dotenv import load_dotenv
import uuid
import mediapipe as mp
import logging
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time 
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#gemini-2.5-pro-exp-03-25
geminimodel = genai.GenerativeModel(model_name="gemini-2.5-pro-exp-03-25")
model = SentenceTransformer("all-MiniLM-L6-v2")

app = Flask(__name__)
app.secret_key = 'dev_key_123'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
def recommendTopics(topic):
    final_df = pd.read_csv('upsc_real_subtopics.csv')
    test = final_df["Subtopics"].apply(lambda text: str(text).lower())
    texts = test.to_list()
    if os.path.exists('plagarism.index'):
        index = faiss.read_index('plagarism.index')
        query_embedding = model.encode([topic], convert_to_tensor=False)
        top_k = 3
        distances, indices = index.search(query_embedding, top_k)
        recommended_texts = [texts[i] for i in indices[0]]
        print("Recommended texts:", recommended_texts)
        return recommended_texts
    else:
        embeddings = model.encode(texts, convert_to_tensor=False)
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings)
        faiss.write_index(index, 'plagarism.index')
        index = faiss.read_index('plagarism.index')
        query_embedding = model.encode([topic], convert_to_tensor=False)
        top_k = 3
        distances, indices = index.search(query_embedding, top_k)
        recommended_texts = [texts[i] for i in indices[0]]
        print("Recommended texts:", recommended_texts)
        return recommended_texts

def generate_questions(topic):
    prompt = f"""You are an expert in {topic}. Your task is to generate 10 multiple-choice questions (MCQs) covering diverse topics within this subject.
    Each question must be followed by four answer choices (a, b, c, d), and the correct answer must be among them.
    Output Format:
    Q1: [Question]?
    a) [Option 1]
    b) [Option 2]
    c) [Option 3]
    d) [Option 4]
    Answer List: ['Option 1', 'Option 2', ..., 'Option 10'] (Correct options exactly as written)
    """
    response = geminimodel.generate_content(prompt)
    text = response.parts[0].text if response.parts else ""

    questions = re.findall(r'Q\d+: (.*?)\?', text)
    options = re.findall(r'a\) (.*?)\nb\) (.*?)\nc\) (.*?)\nd\) (.*?)\n', text)
    match = re.search(r"Answer List:\s*(\[[^\]]*\])", text)
    correct_answers = ast.literal_eval(match.group(1)) if match else []

    questions = questions[:10] + ["Question Missing"] * (10 - len(questions))
    options = options[:10] + [("", "", "", "")] * (10 - len(options))
    correct_answers = correct_answers[:10] + [""] * (10 - len(correct_answers))
    #print(questions)
    count =0
    for q in questions:
        if q!="Question Missing":
            count=count+1
    #questions=questions[:count]
    #print(questions)
    
    #print(options)
    #print(count)
    questions=questions[:count]
    options=options[:count]
    correct_answers=correct_answers[:count]
    print("correct answers  in generate questions",correct_answers)
    question_data = []
    for i, (q, opt, ans) in enumerate(zip_longest(questions, options, correct_answers, fillvalue="")):
        formatted_options = list(opt) if opt else ["", "", "", ""]
        corrected_answer = ans.strip()

        if corrected_answer not in formatted_options:
            corrected_answer = formatted_options[0]

        question_data.append({
            "question": q,
            "options": formatted_options,
            "correct": corrected_answer
        })

    return question_data,count

def get_subtopic(question):
    prompt = f"Classify the following question into a single subtopic:\n\n{question}\n\nJust return the subtopic name."
    response = geminimodel.generate_content(prompt)
    if response and response.parts:
        return response.parts[0].text.strip()
    return "Unknown Subtopic"



BOOKS = [
    {"name": "Harrisons Rheumatology", "book_path": "Harrisons_Rheumatology.pdf", "embeddings_path": "Harrisons_Rheumatology.index"},
    {"name": "Harrisons Nephrology And Acid Base", "book_path": "Harrisons_Nephrology_and_Acid_Base.pdf", "embeddings_path": "Harrisons_Nephrology_and_Acid_Base.index"},
    {"name": "Harrisons Manual Of Medicine", "book_path": "Harrisons_Manual_Of_Medicine.pdf", "embeddings_path": "Harrisons_Manual_Of_Medicine.index"},
    {"name": "Harrisons Manual Of Oncology", "book_path": "Harrisons_Manual_of_Oncology.pdf", "embeddings_path": "Harrisons_Manual_of_Oncology.index"},
    {"name": "Harrisons Pulmonary And Critical Care Medicine", "book_path": "Harrisons_Pulmonary_and_Critical_Care_Medicine.pdf", "embeddings_path": "Harrisons_Pulmonary_and_Critical_Care_Medicine.index"},
    {"name": "Harrisons Cardio Vascular Medicine ", "book_path": "harrisons_cardiovascular_medicine.pdf", "embeddings_path": "harrisons_cardiovascular_medicine.index"},
    {"name": "Harrisons Endocrinology", "book_path": "harrisons_endocrinology.pdf", "embeddings_path": "harrisons_endocrinology.index"},
    {"name": "Harrisons Gastroenterology And Hepatology", "book_path": "harrisons_gastroenterology_and_hepatology.pdf", "embeddings_path": "harrisons_gastroenterology_and_hepatology.index"},
    {"name": "Harrisons Hematology And Oncology", "book_path": "harrisons_hematology_and_oncology.pdf", "embeddings_path": "harrisons_hematology_and_oncology.index"},
    {"name": "Harrisons Manual Of Medicine", "book_path": "Harrisons_Manual_Of_Medicine.pdf", "embeddings_path": "Harrisons_Manual_Of_Medicine.index"},
    {"name": "Harrisons Nephrology And Acid Base", "book_path": "Harrisons_Nephrology_and_Acid_Base.pdf", "embeddings_path": "Harrisons_Nephrology_and_Acid_Base.index"},
    {"name": "Harrisons Neurology In Clinical Medicine", "book_path": "Harrisons_neurology_in_clinical_medicine.pdf", "embeddings_path": "Harrisons_neurology_in_clinical_medicine.index"},
    {"name":"PathologyRobbins","book_path":"PathologyRobbins7ed.pdf","embeddings_path":"PathologyRobbins7ed.index"},
    {"name":"KD Tripathi Essentials of Medical Pharmacology","book_path":"KD_Tripathi_Essentials_of_Medical_PharmacologyUnitedVRG_2013.pdf","embeddings_path":"PathologyRobbins7ed.index"},
    {"name":"Harpers Illustrated Biochemistry","book_path":"Harpers_Illustrated_Biochemistry_Thirty_Second_Edition _2023.pdf","embeddings_path":"Harpers_Illustrated_Biochemistry_Thirty_Second_Edition_2023.index"},
    {"name":"Gray Anatomy","book_path":"Grays_Anatomy-41_E.pdf","embeddings_path":"Grays_Anatomy-41_E.index"}
]
@app.route('/book_test')
def books_name():
    return render_template('book_dropdown.html', books=BOOKS)
@app.route('/get_paths', methods=['POST'])
def get_paths():
    book_name = request.form.get('book_name')
    #print("query is ",query)
    print(f"Received book name: {book_name}") 
    for i in range(0,len(BOOKS)):
        if BOOKS[i]["name"]==book_name:
            book_path=BOOKS[i]["book_path"]
            book_embeddings=BOOKS[i]["embeddings_path"]
            def load_pdf_and_split(pdf_path):
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(documents)
                print("loaded_pdf_file_and chunks_created")
                return chunks
            def generate_embeddings(chunks):
                model = SentenceTransformer('all-MiniLM-L6-v2')
                texts = [chunk.page_content for chunk in chunks]
                embeddings = model.encode(texts, convert_to_numpy=True)
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)
                faiss.write_index(index, 'hello')
                print("generate embeddings")
                return index
            def load_embeddings():
                index = faiss.read_index(book_embeddings)
                print("Sucessfully loaded embeddings")
                return index
            def retrieve_similar_chunks(index, query, chunks, top_k=3):
                model = SentenceTransformer('all-MiniLM-L6-v2')
                query_embedding = model.encode([query], convert_to_numpy=True)
                distances, indices = index.search(query_embedding, top_k)
                similar_chunks = [(chunks[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
                print("retrieve_similar_chunks")
                return similar_chunks
            def generate_quetions(Retrived_text,query):
                prompt = f"""
                use the {Retrived_text} use it to generate challenging MCQ'S and make sure that the question created are related to {query} .
                Each question must be followed by four answer choices (a, b, c, d), and the correct answer must be among them.please generate 5 questions only.
                make sure that answers are present in the last in the form of a list.please make the questions and mcq's to be precise.
                give me the hard questions and needs test the depth knowledge of the user. please be on the point 
                Output Format:
                Q1: [Question]?
                a) [Option 1]
                b) [Option 2]
                c) [Option 3]
                d) [Option 4]
                Answer List: ['Option 1', 'Option 2', ..., 'Option 4'] (Correct options exactly as written)
                """
                response = geminimodel.generate_content(prompt)
                text = response.parts[0].text if response.parts else ""
                #print(text)
                questions = re.findall(r'Q\d+: (.*?)\?', text)
                options = re.findall(r'a\) (.*?)\nb\) (.*?)\nc\) (.*?)\nd\) (.*?)\n', text)
                match = re.search(r"Answer List:\s*(\[[^\]]*\])", text)
                correct_answers = ast.literal_eval(match.group(1)) if match else []
                print("generating_question")
                print("-----------------------------------------------------------------------")
                print(questions)
                print("-----------------------------------------------------------------------")
                print(options)
                print("------------------------------------------------------------------------")
                print(correct_answers)
                question_data = []
                for i, (q, opt, ans) in enumerate(zip_longest(questions, options, correct_answers, fillvalue="")):
                    formatted_options = list(opt) if opt else ["", "", "", ""]
                    corrected_answer = ans.strip()
                    print(corrected_answer)
                    if corrected_answer not in formatted_options:
                        corrected_answer = formatted_options[0]

                    question_data.append({
                        "question": q,
                        "options": formatted_options,
                        "correct": corrected_answer
                    })

                return question_data 
            Retrived_text=[]
            chunks = load_pdf_and_split(book_path)
            if os.path.exists(book_path):
              index=load_embeddings()
            else:
              index=generate_embeddings(chunks)
            query="Transdermal therapeutic systems (TTS)"
            similar_chunks = retrieve_similar_chunks(index, query, chunks)
            for i, (chunk, distance) in enumerate(similar_chunks):
              Retrived_text.append(chunk.page_content)
            #print(Retrived_text)
            #print("retrieve_similar_chunks")
            question_data=generate_quetions(Retrived_text,query)
            #print(question_data)
            return render_template("books_questions.html",questions_data=question_data)
    return '', 204 


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        topic = request.form['topic']
        questions,count = generate_questions(topic)
        return render_template('quiz.html', questions=questions, topic=topic,count=count)
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    print("iam in submit route")

    questions = ast.literal_eval(request.form['questions'])
    #options= ast.literal_eval(request.form['options'])
    print(questions)
    weak_questions = []
    weak_topics = []
    score = 0
    wrong_options=[]
    total = len(questions)
    for i, question in enumerate(questions):
        selected = request.form.get(f'question{i}', '').strip()
        correct_answer = question['correct'].strip()
        print("-------------------------------------")
        print(correct_answer)
        print("------------------------------------")
        print(selected)
        if selected == correct_answer:
            score += 1
        else:
            wrong_options.append(questions[i]['options'])
            weak_questions.append(question['question'])
    for q in weak_questions:
        weak_topics.append(get_subtopic(q))
    
    articles_list = []
    """
    print("-------------------------------------------------------------------------------")
    print(wrong_options)
    print("-------------------------------------------------------------------------------")

    for weak_topic in weak_topics:
        params = {
            "q": weak_topic,
            "hl": "en",
            "api_key": os.getenv("SERP_API_KEY")
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        if "organic_results" in results:
            for result in results["organic_results"][:3]:
                articles_list.append({
                    "question": weak_topic,
                    "recommended link": result['link'],
                })
    """
    explanation_list=[]
    print("len of weak question ",len(weak_questions),"wrong options len ",len(wrong_options))
    for i,weak in enumerate(weak_questions):
        prompt=f"""provide conisice and crisp explanation answer for the {weak} question from the following options {wrong_options[i]} """
        response = geminimodel.generate_content(prompt)
        text = response.parts[0].text if response.parts else ""
        explanation_list.append(text)
    question_explanations = list(zip(weak_questions, explanation_list))
    return render_template('result.html', score=score, total=total, weak_topics=weak_topics, articles=articles_list,question_explanations=question_explanations)

@app.route('/recommendations', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        topic = request.form['topic']
        recommendedTopics = recommendTopics(topic)
        #print(recommendedTopics)
        return render_template('recommend.html', recommendedTopics=recommendedTopics, topic=topic)
    return render_template('recommendationsForm.html')

"""
@app.route('/chatbuddy', methods=["GET", "POST"])
def chatting():
    if request.method == "POST":
        topic = request.form["topic"]
        session["topic"] = topic
        session["chat"] = geminimodel.start_chat()
        session["history"] = []
        return redirect(url_for("chat"))
    return render_template("chatting_template.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "topic" not in session or "chat" not in session:
        return redirect(url_for("chatting"))

    topic = session["topic"]
    chat_obj = session["chat"]
    history = session.get("history", [])

    if request.method == "POST":
        user_input = request.form["message"]
        if user_input.lower() == "exit":
            return redirect(url_for("reset"))
        
        prompt = f"You are an expert in {topic}.
Please give me the answer for the {user_input} in concise and clean manner.
Make sure to give the response in same font."

        response = chat_obj.send_message(prompt)
        bot_reply = response.text

        history.append(("You", user_input))
        history.append(("Gemini", bot_reply))
        session["history"] = history

    return render_template("chat.html", topic=topic, history=history)

@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for("chatting"))
"""
chat_sessions = {} 
@app.route('/chatbuddy', methods=["GET", "POST"])
def chatting():
    if request.method == "POST":
        topic = request.form["topic"]
        chat_id = str(uuid.uuid4())  
        chat_sessions[chat_id] = geminimodel.start_chat() 
        session["topic"] = topic
        session["chat_id"] = chat_id
        session["history"] = []

        return redirect(url_for("chat"))
    return render_template("chatting_template.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "topic" not in session or "chat_id" not in session:
        return redirect(url_for("chatting"))

    topic = session["topic"]
    chat_id = session["chat_id"]
    chat_obj = chat_sessions.get(chat_id)

    if not chat_obj:
        return redirect(url_for("reset"))  
    history = session.get("history", [])

    if request.method == "POST":
        user_input = request.form["message"]
        if user_input.lower() == "exit":
            return redirect(url_for("reset"))

        prompt = f"""You are an expert in {topic}.
Please give me the answer for the {user_input} in concise and clean manner.
Make sure to give the response in same font."""

        response = chat_obj.send_message(prompt)
        bot_reply = response.text

        history.append(("You", user_input))
        history.append(("Gemini", bot_reply))
        session["history"] = history

    return render_template("chat.html", topic=topic, history=history)

@app.route("/reset")
def reset():
    chat_id = session.get("chat_id")
    if chat_id and chat_id in chat_sessions:
        del chat_sessions[chat_id]  
    session.clear()
    return redirect(url_for("chatting"))


if __name__ == '__main__':
    app.run(debug=True)