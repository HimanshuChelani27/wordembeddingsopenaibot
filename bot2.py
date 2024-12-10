import sys
import os
import requests
from flask import Flask, request, jsonify
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from flask_cors import CORS
from docx import Document  # Import the docx library to read Word files

chat_bot = Flask(__name__)
CORS(chat_bot)  # Enable CORS

user_name = "You"
bot_name = "AgriBot"
conversation = ""

# Function to read contents of a Word document
def read_word_file(file_path):
    """Reads the content of a Word file and returns the text as a string."""
    try:
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip() != '']  # Exclude empty paragraphs
        return "\n".join(paragraphs)
    except Exception as e:
        return f"Error reading Word file: {str(e)}"


def build_final_result():
    """Build the final result using the various append functions."""
    final_result = []

    # Path to the Word document
    word_file_path = "Department of Drinking Water and Sanitation.docx"  # Replace with your Word file path

    # Read the Word file content and append it to final_result
    word_content = read_word_file(word_file_path)
    
    final_result.append(word_content)

    return final_result


@chat_bot.route('/assistants', methods=['POST'])
def receive_query():
    try:
        data = request.get_json()
        user_question = request.json.get('question')

        # Prepare the final result
        final_result = build_final_result()
        print("final_result----------------->", final_result)


        # If OpenAI LLM usage is necessary
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=5000, chunk_overlap=500, length_function=len)
        chunks = text_splitter.split_text("\n".join(final_result))

        # Use the OpenAI API key explicitly here
        openai_api_key = "your_open_ai_key"  # Replace with your actual OpenAI API key
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Pass the API key here
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        if user_question:
            predefined_responses = {
                "what is agribot?": "Agribot is an AI-powered agricultural assistant developed by Agripilot.ai."
            }

            response = predefined_responses.get(user_question.lower())
            if response:
                return jsonify(response)
                

            # Use OpenAI LLM to answer the question
            # full_question = f"This is my question: {user_question} Give response in points"

            instruction = (
                "Please answer the following question using only the information provided in the document. "
                "Do not use any other knowledge. "
                "If the answer is not found in the provided data, simply say 'I don't know'."
            )
            
            full_question = f"{instruction} Question: {user_question}"
            docs = knowledge_base.similarity_search(full_question)
            llm = OpenAI(openai_api_key=openai_api_key)  # Pass the API key here
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=full_question, model="gpt-4-1106-preview")

            return jsonify(response)

        return jsonify({"error": "Please try again later, Agribot's traffic is busy"})

    except KeyError as ke:
        response = {"error": f"Missing key: {str(ke)}"}
    except Exception as e:
        response = {"error": str(e)}

    return jsonify(response)


if __name__ == '__main__':
    chat_bot.run(host='0.0.0.0', port=8083, debug=True)
