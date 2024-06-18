import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Milvus
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer, util
import os
from PIL import Image
import io

# Load environment variables
load_dotenv()

def get_db_connection(collection_name):
    """Returns a Milvus DB connection object"""
    embeddings = OpenAIEmbeddings()
    return Milvus(embeddings, connection_args={
        "user": os.getenv("MILVUS_DB_USERNAME"),
        "password": os.getenv("MILVUS_DB_PASSWORD"),
        "host": os.getenv("MILVUS_DB_HOST"),
        "port": os.getenv("MILVUS_DB_PORT"),
        "db_name": os.getenv("MILVUS_DB_NAME")},
                  collection_name=os.getenv("MILVUS_DB_COLLECTION"))

def get_similar_docs(query: str):
    """Fetches similar text from the vector db"""
    vector_db = get_db_connection("my_collection")
    return vector_db.similarity_search_with_score(query, k=3)

def fetch_answer_from_llm(query: str):
    """Fetches relevant answer from LLM"""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                     temperature=0.6,
                     max_tokens=1024)
    chain = load_qa_chain(llm, "stuff")
    similar_docs = get_similar_docs(query)
    docs = []
    for doc in similar_docs:
        docs.append(doc[0])
    chain_response = chain.invoke(input={"input_documents": docs, "question": query})
    return chain_response["output_text"]

def find_relevant_images(alarm_description, images_folder='./extracted_figures', top_k=10):
    """Finds relevant images related to the alarm description using similarity search"""
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    alarm_embedding = model.encode(alarm_description, convert_to_tensor=True)
    
    relevant_images = []
    image_titles = []
    similarities = []

    for filename in os.listdir(images_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            title_path = os.path.join(images_folder, filename.replace('.jpg', '_title.txt').replace('.jpeg', '_title.txt').replace('.png', '_title.txt'))
            if os.path.exists(title_path):
                with open(title_path, 'r') as file:
                    title = file.read().strip()
                    title_embedding = model.encode(title, convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(alarm_embedding, title_embedding).item()
                    similarities.append((similarity, filename, title))
    
    # Sort by similarity and select top_k
    similarities.sort(reverse=True, key=lambda x: x[0])
    for sim, filename, title in similarities[:top_k]:
        relevant_images.append(os.path.join(images_folder, filename))
        image_titles.append(title)
    
    return relevant_images, image_titles

def load_troubleshoot_data(file_path):
    """Load the TroubleShoot Excel file and preprocess"""
    df = pd.read_excel(file_path)
    df['combined_info'] = df['Error'] + ' ' + df['System'] + ' ' + df['Causes']
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    df['embedding'] = df['combined_info'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

def search_troubleshoot_data(df, query):
    """Search the TroubleShoot data for the most similar entry"""
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode(query, convert_to_tensor=True)
    df['similarity'] = df['embedding'].apply(lambda x: util.pytorch_cos_sim(query_embedding, x).item())
    best_match = df.loc[df['similarity'].idxmax()]
    return best_match['INFO (Section/Work Card in DO0F466162E Maintenance Manual)'], best_match['Code']

def generate_chain_of_answers(alarm_id, description, subsystem, troubleshoot_df):
    """Generates a chain of answers for the provided input values"""
    try:
        questions = [
            "Sensor(s) the alarm is related to.",
            "Sensor measuring point(s).",
            "Sensor description.",
            "Sensor measuring range.",
            "System the sensor is related to.",
            "System the sensor is dependent on.",
            "Related sensors to given sensor.",
            "Potential triggers of the alarm:",
            "What possible actions could a crew member take given the alarm? Please provide the Section/Work Card in DO0F466162E Maintenance Manual if possible",
            "What technical knowledge is required to solve the problem?",
            "What parts (if any) would be required to solve the problem?",
            "What drawings or diagrams (if any) are present in the technical documentation that relate to this alarm?"
        ]

        context = f"I received an error code: {alarm_id}. I know from experience that it's a {description} at {subsystem} subsystem."
        responses = {}
        for question in questions:
            if question == "What possible actions could a crew member take given the alarm? Please provide the Section/Work Card in DO0F466162E Maintenance Manual if possible":
                query = f"{context} Please provide answer to the question {question}"
                answer = fetch_answer_from_llm(query)
                info, code = search_troubleshoot_data(troubleshoot_df, query)
                answer += f" INFO: {info}, Code: {code}"
            else:
                query = f"{context} Please provide answer to the question {question}"
                answer = fetch_answer_from_llm(query)
            
            responses[question] = answer
            context += f" {answer}"
            st.markdown(f"<div style='color: #0056b3; font-weight: bold;'>**Question:** {question}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color: #28a745;'>**Answer:** {answer}</div>", unsafe_allow_html=True)

        # For the last question, find and display relevant images
        if "What drawings or diagrams (if any) are present in the technical documentation that relate to this alarm?" in questions:
            relevant_images, image_titles = find_relevant_images(description)
            st.markdown("<div style='color: #0056b3; font-weight: bold;'>**Relevant Drawings or Diagrams:**</div>", unsafe_allow_html=True)
            if relevant_images:
                for image_path, title in zip(relevant_images, image_titles):
                    st.image(image_path, caption=title)
            else:
                st.markdown("<div style='color: #dc3545;'>No relevant drawings or diagrams found.</div>", unsafe_allow_html=True)

        return responses

    except Exception as exception_message:
        st.markdown(f"<div style='color: #dc3545;'>Error: {str(exception_message)}</div>", unsafe_allow_html=True)

def main():
    st.title("🔔 Alarm Query System")

    st.write("Enter the details of the alarm to get the answers to the related questions:")

    if "responses" not in st.session_state:
        st.session_state.responses = None

    alarm_id = st.text_input("🚨 ALARM_ID")
    description = st.text_input("📝 DESCRIPTION")
    subsystem = st.text_input("🛠️ SUBSYSTEM")

    troubleshoot_file_path = "./TroubleShoot.xlsx"  # Update this path to the location of your TroubleShoot Excel file
    troubleshoot_df = load_troubleshoot_data(troubleshoot_file_path)

    if st.button("Get Answers"):
        if alarm_id and description and subsystem:
            st.session_state.responses = generate_chain_of_answers(alarm_id, description, subsystem, troubleshoot_df)
        else:
            st.markdown("<div style='color: #dc3545;'>Please fill in all the fields.</div>", unsafe_allow_html=True)

    if st.button("Reset"):
        st.session_state.responses = None
        st.experimental_rerun()

    # if st.session_state.responses:
    #     st.markdown("<div style='color: #17a2b8;'>Results:</div>", unsafe_allow_html=True)
    #     for question, answer in st.session_state.responses.items():
    #         st.markdown(f"<div style='color: #0056b3; font-weight: bold;'>**Question:** {question}</div>", unsafe_allow_html=True)
    #         st.markdown(f"<div style='color: #28a745;'>**Answer:** {answer}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
