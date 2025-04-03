import os
import tempfile
import sqlite3
import uuid
import hashlib
from datetime import datetime
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document

# Constants
DATA_DIR = "./data"
DB_PATH = os.path.join(DATA_DIR, "study_assistant.db")
os.makedirs(DATA_DIR, exist_ok=True)
 
# Database initialization with streamlined tables
def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.executescript('''
            CREATE TABLE IF NOT EXISTS pdfs (
                id TEXT PRIMARY KEY,
                filename TEXT,
                hash TEXT,
                upload_date TIMESTAMP,
                vector_store_path TEXT,
                page_count INTEGER DEFAULT 0,
                has_images INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS queries (
                id TEXT PRIMARY KEY,
                pdf_id TEXT,
                query TEXT,
                response TEXT,
                timestamp TIMESTAMP,
                is_important BOOLEAN DEFAULT 0,
                category TEXT,
                FOREIGN KEY (pdf_id) REFERENCES pdfs (id)
            );
            CREATE TABLE IF NOT EXISTS flashcards (
                id TEXT PRIMARY KEY,
                pdf_id TEXT,
                question TEXT,
                answer TEXT,
                created_date TIMESTAMP,
                last_reviewed TIMESTAMP,
                confidence_level INTEGER DEFAULT 1,
                FOREIGN KEY (pdf_id) REFERENCES pdfs (id)
            );
            CREATE TABLE IF NOT EXISTS study_notes (
                id TEXT PRIMARY KEY,
                pdf_id TEXT,
                topic TEXT,
                content TEXT,
                created_date TIMESTAMP,
                FOREIGN KEY (pdf_id) REFERENCES pdfs (id)
            );
            CREATE TABLE IF NOT EXISTS pdf_images (
                id TEXT PRIMARY KEY,
                pdf_id TEXT,
                page_num INTEGER,
                image_path TEXT,
                extraction_date TIMESTAMP,
                FOREIGN KEY (pdf_id) REFERENCES pdfs (id)
            );
        ''')

        # Ensure the necessary columns exist in the pdfs table
        try:
            cursor.execute("ALTER TABLE pdfs ADD COLUMN page_count INTEGER DEFAULT 0;")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE pdfs ADD COLUMN has_images INTEGER DEFAULT 0;")
        except sqlite3.OperationalError:
            pass  # Column already exists

        conn.commit()
        return conn
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return None

# Get table schema for debugging
def get_table_info(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()

# Configure API
@st.cache_resource
def configure_api():
    api_key = os.getenv("GOOGLE_API_KEY") 
    if not api_key:
        api_key = st.sidebar.text_input("Enter Google API Key:", type="password")
        if not api_key:
            st.warning("Please provide a Google API Key to continue.")
            st.stop()
    genai.configure(api_key=api_key)
    return api_key

# Helper functions
@st.cache_resource
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def compute_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

@st.cache_data
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_per_page = [(page_num, page.get_text("text")) for page_num, page in enumerate(doc)]
    page_count = len(doc)
    has_images = any(page.get_images() for page in doc)
    doc.close()
    return text_per_page, page_count, has_images

def extract_and_save_all_images(pdf_path, pdf_id, conn):
    doc = fitz.open(pdf_path)
    cursor = conn.cursor()
    
    images_saved = 0
    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            
            # Create a unique filename for the image
            image_id = str(uuid.uuid4())
            img_filename = f"img_{pdf_id}_{page_num}_{img_index}.png"
            img_path = os.path.join(DATA_DIR, img_filename)
            
            # Save the image to file
            with open(img_path, "wb") as img_file:
                img_file.write(base_image["image"])
            
            # Record the image in the database
            cursor.execute("INSERT INTO pdf_images VALUES (?, ?, ?, ?, ?)",
                          (image_id, pdf_id, page_num, img_path, datetime.now()))
            images_saved += 1
    
    conn.commit()
    doc.close()
    return images_saved

def extract_images_from_pages(pdf_path, page_nums):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in page_nums:
        if page_num < len(doc):
            page = doc[page_num]
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=DATA_DIR) as tmp:
                    tmp.write(base_image["image"])
                    images.append(tmp.name)
    doc.close()
    return images

def get_relevant_images(pdf_id, page_nums, conn):
    cursor = conn.cursor()
    images = []
    
    for page_num in page_nums:
        cursor.execute("SELECT image_path FROM pdf_images WHERE pdf_id = ? AND page_num = ?", 
                      (pdf_id, page_num))
        results = cursor.fetchall()
        images.extend([img_path[0] for img_path in results])
    
    return images

# Index PDF text
def index_pdf_text(text_per_page, pdf_id):
    documents = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for page_num, text in text_per_page:
        chunks = text_splitter.split_text(text)
        documents.extend(Document(page_content=chunk, metadata={'page': page_num, 'pdf_id': pdf_id}) for chunk in chunks)
    
    vector_store_path = os.path.join(DATA_DIR, f"vector_store_{pdf_id}")
    vector_store = FAISS.from_documents(documents, get_embedding_function())
    vector_store.save_local(vector_store_path)
    return vector_store, vector_store_path

# Load vector store safely
def load_vector_store(path):
    try:
        return FAISS.load_local(path, get_embedding_function(), allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

# Optimized Gemini API querying
def query_gemini(prompt, context, detail_level="medium", query_type="normal"):
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Simplified prompt templates
    prompt_templates = {
        "normal": f"Context: {context}\n\nQuestion: {prompt}\n\nProvide a {detail_level} answer based ONLY on the context.",
        "explain": f"Context: {context}\n\nConcept to explain: {prompt}\n\nProvide a {detail_level} explanation with examples.",
        "example": f"Context: {context}\n\nConcept: {prompt}\n\nProvide {detail_level} examples of this concept.",
        "summarize": f"Context: {context}\n\nTopic to summarize: {prompt}\n\nProvide a {detail_level} summary.",
        "quiz": f"Context: {context}\n\nTopic for quiz: {prompt}\n\nGenerate a {detail_level} quiz with answers."
    }
    
    try:
        response = model.generate_content(prompt_templates.get(query_type, prompt_templates["normal"]))
        return response.text
    except Exception as e:
        return f"Error querying AI: {e}"

# Search PDF and answer
def search_pdf_and_answer(query, vector_store, pdf_id, conn, detail_level, query_type):
    docs = vector_store.similarity_search(query, k=4)
    context = "\n".join(doc.page_content for doc in docs)
    
    answer = query_gemini(query, context, detail_level, query_type)
    
    # Save query to history
    cursor = conn.cursor()
    query_id = str(uuid.uuid4())
    cursor.execute("INSERT INTO queries VALUES (?, ?, ?, ?, ?, ?, ?)", 
                  (query_id, pdf_id, query, answer, datetime.now(), False, query_type))
    conn.commit()
    
    page_nums = set(doc.metadata['page'] for doc in docs)
    images = get_relevant_images(pdf_id, page_nums, conn)
    return answer, images, [doc.page_content for doc in docs], query_id

# Create flashcard from query
def create_flashcard(query, answer, pdf_id, conn):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO flashcards VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (str(uuid.uuid4()), pdf_id, query, answer, datetime.now(), datetime.now(), 1))
    conn.commit()

# Mark query as important
def mark_important(query_id, conn):
    cursor = conn.cursor()
    cursor.execute("UPDATE queries SET is_important = 1 WHERE id = ?", (query_id,))
    conn.commit()

# Generate study notes for a topic
def generate_study_notes(topic, vector_store, pdf_id, conn):
    docs = vector_store.similarity_search(f"information about {topic}", k=8)
    context = "\n".join(doc.page_content for doc in docs)
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    study_prompt = f"""
    Based on the following context about "{topic}", create comprehensive study notes that include:
    1. Key definitions and concepts
    2. Important formulas or principles
    3. Examples that demonstrate the concepts
    4. Summary of main points

    Format these as structured study notes.
    Context: {context}
    """
    
    try:
        response = model.generate_content(study_prompt)
        notes = response.text
        
        # Save study notes
        cursor = conn.cursor()
        cursor.execute("INSERT INTO study_notes VALUES (?, ?, ?, ?, ?)",
                      (str(uuid.uuid4()), pdf_id, topic, notes, datetime.now()))
        conn.commit()
        
        return notes
    except Exception as e:
        return f"Error generating study notes: {e}"

# Streamlit UI with optimized layout
def main():
    st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="wide")
    
    st.title("📚 Quick Study Assistant")
    
    # Initialize session state
    if "pdf_path" not in st.session_state:
        st.session_state.pdf_path = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "pdf_id" not in st.session_state:
        st.session_state.pdf_id = None
    if "current_query_id" not in st.session_state:
        st.session_state.current_query_id = None
        
    # Sidebar settings with simplified options
    with st.sidebar:
        st.header("Settings")
        
        detail_level = st.radio(
            "Answer Detail:",
            ["short", "medium", "detailed"],
            index=1
        )
        
        query_type = st.selectbox(
            "Query Type:",
            ["normal", "explain", "example", "summarize", "quiz"],
            index=0
        )
        
        # Configure API Key
        api_key = configure_api()
    
    # Main layout with simplified tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Study", "Flashcards", "Study Notes", "History"])
    
    conn = init_db()
    if not conn:
        st.error("Failed to initialize database. Please restart the application.")
        return
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Study Material")
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            
            if uploaded_file:
                with st.spinner("Processing PDF..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=DATA_DIR) as tmp:
                        tmp.write(uploaded_file.read())
                        pdf_path = tmp.name
                    
                    file_hash = compute_file_hash(pdf_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, vector_store_path FROM pdfs WHERE hash = ?", (file_hash,))
                    existing = cursor.fetchone()
                    
                    if existing and os.path.exists(existing[1]):
                        st.session_state.pdf_id, vector_store_path = existing[0], existing[1]
                        st.session_state.vector_store = load_vector_store(vector_store_path)
                        st.session_state.pdf_path = pdf_path
                        st.success("Study material loaded! Ready to study.")
                    else:
                        st.session_state.pdf_id = str(uuid.uuid4())
                        text_per_page, page_count, has_images = extract_text_from_pdf(pdf_path)
                        st.session_state.vector_store, vector_store_path = index_pdf_text(text_per_page, st.session_state.pdf_id)
                        st.session_state.pdf_path = pdf_path
                        
                        # Add debugging to check values before insertion
                        try:
                            # Check if connection is still valid
                            conn.execute("SELECT 1")
                            
                            # Print data types for debugging (you can remove this later)
                            print(f"PDF ID: {st.session_state.pdf_id}, Type: {type(st.session_state.pdf_id)}")
                            print(f"Filename: {uploaded_file.name}, Type: {type(uploaded_file.name)}")
                            print(f"Hash: {file_hash}, Type: {type(file_hash)}")
                            print(f"Vector Store Path: {vector_store_path}, Type: {type(vector_store_path)}")
                            print(f"Page Count: {page_count}, Type: {type(page_count)}")
                            print(f"Has Images: {has_images}, Type: {type(has_images)}")
                            
                            # Use a direct approach with explicit column names
                            cursor.execute("""
                                INSERT INTO pdfs 
                                (id, filename, hash, upload_date, vector_store_path, page_count, has_images) 
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                st.session_state.pdf_id,
                                uploaded_file.name,
                                file_hash,
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Format the datetime
                                vector_store_path,
                                page_count,
                                1 if has_images else 0  # Ensure boolean is 0 or 1 for SQLite
                            ))
                            conn.commit()
                            
                            # Extract and save all images from the PDF
                            with st.spinner("Extracting images from PDF..."):
                                images_count = extract_and_save_all_images(pdf_path, st.session_state.pdf_id, conn)
                                if images_count > 0:
                                    st.success(f"Extracted {images_count} images from the PDF.")
                            
                            st.success("Study material indexed! Ready to study.")
                        except sqlite3.Error as e:
                            st.error(f"Database error: {e}")
                            # Try to debug table schema
                            try:
                                table_info = get_table_info(conn, "pdfs")
                                st.error(f"Table schema: {table_info}")
                            except:
                                pass
        
        with col2:
            st.subheader("Ask Questions")
            if st.session_state.vector_store and st.session_state.pdf_path:
                query = st.text_input("Your study question:")
                
                if query:
                    with st.spinner("Searching..."):
                        answer, images, source_texts, query_id = search_pdf_and_answer(
                            query, 
                            st.session_state.vector_store, 
                            st.session_state.pdf_id, 
                            conn,
                            detail_level,
                            query_type
                        )
                        st.session_state.current_query_id = query_id
                    
                    st.markdown(f"### Answer")
                    st.write(answer)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Add to Flashcards"):
                            create_flashcard(query, answer, st.session_state.pdf_id, conn)
                            st.success("Added to flashcards!")
                    with col2:
                        if st.button("Mark as Important"):
                            mark_important(st.session_state.current_query_id, conn)
                            st.success("Marked as important!")
                    
                    with st.expander("View source context"):
                        for i, text in enumerate(source_texts):
                            st.markdown(f"**Source {i+1}:**")
                            st.write(text)
                    
                    if images:
                        with st.expander("View relevant images", expanded=True):
                            cols = st.columns(min(3, len(images)))
                            for i, img in enumerate(images):
                                cols[i % 3].image(img, use_container_width=True)
                    else:
                        st.info("No relevant images found for this query.")
            else:
                st.info("Please upload study material to begin")
    
    # Flashcards tab
    with tab2:
        st.subheader("Flashcards")
        if st.session_state.pdf_id:
            cursor = conn.cursor()
            cursor.execute("SELECT id, question, answer, confidence_level FROM flashcards WHERE pdf_id = ? ORDER BY last_reviewed ASC", 
                          (st.session_state.pdf_id,))
            flashcards = cursor.fetchall()
            
            if flashcards:
                st.write(f"{len(flashcards)} flashcards available")
                
                # Simple flashcard review system
                if st.button("Study Flashcards"):
                    st.session_state.reviewing_flashcards = True
                    st.session_state.current_flashcard_index = 0
                
                if "reviewing_flashcards" in st.session_state and st.session_state.reviewing_flashcards:
                    if st.session_state.current_flashcard_index < len(flashcards):
                        flashcard = flashcards[st.session_state.current_flashcard_index]
                        
                        st.write(f"**Flashcard {st.session_state.current_flashcard_index + 1}/{len(flashcards)}**")
                        st.write(f"**Question:** {flashcard[1]}")
                        
                        show_answer = st.button("Show Answer")
                        if show_answer:
                            st.write(f"**Answer:** {flashcard[2]}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("Easy"):
                                    cursor.execute("UPDATE flashcards SET confidence_level = ?, last_reviewed = ? WHERE id = ?",
                                                 (min(flashcard[3] + 1, 5), datetime.now(), flashcard[0]))
                                    conn.commit()
                                    st.session_state.current_flashcard_index += 1
                            with col2:
                                if st.button("Medium"):
                                    cursor.execute("UPDATE flashcards SET last_reviewed = ? WHERE id = ?",
                                                 (datetime.now(), flashcard[0]))
                                    conn.commit()
                                    st.session_state.current_flashcard_index += 1
                            with col3:
                                if st.button("Hard"):
                                    cursor.execute("UPDATE flashcards SET confidence_level = ?, last_reviewed = ? WHERE id = ?",
                                                 (max(flashcard[3] - 1, 1), datetime.now(), flashcard[0]))
                                    conn.commit()
                                    st.session_state.current_flashcard_index += 1
                    else:
                        st.success("Review complete!")
                        st.session_state.reviewing_flashcards = False
            else:
                st.info("No flashcards yet. Add some while studying!")
        else:
            st.info("Upload a document to create flashcards")
    
    # Study Notes tab
    with tab3:
        st.subheader("Study Notes")
        if st.session_state.vector_store and st.session_state.pdf_id:
            topic = st.text_input("Enter a topic:")
            if st.button("Generate Notes"):
                if topic:
                    with st.spinner(f"Generating notes for {topic}..."):
                        notes = generate_study_notes(topic, st.session_state.vector_store, st.session_state.pdf_id, conn)
                    
                    st.markdown("### Study Notes")
                    st.markdown(notes)
                    
                    # Option to download notes
                    notes_filename = f"{topic.replace(' ', '_')}_notes.md"
                    st.download_button(
                        label="Download Notes",
                        data=notes,
                        file_name=notes_filename,
                        mime="text/markdown"
                    )
                else:
                    st.warning("Please enter a topic")
            
            # Past study notes
            cursor = conn.cursor()
            cursor.execute("SELECT id, topic, created_date FROM study_notes WHERE pdf_id = ? ORDER BY created_date DESC", 
                          (st.session_state.pdf_id,))
            notes_list = cursor.fetchall()
            
            if notes_list:
                st.subheader("Previous Notes")
                for note_id, topic, timestamp in notes_list:
                    with st.expander(f"{topic} - {timestamp}"):
                        cursor.execute("SELECT content FROM study_notes WHERE id = ?", (note_id,))
                        content = cursor.fetchone()[0]
                        st.markdown(content)
                        
                        notes_filename = f"{topic.replace(' ', '_')}_notes.md"
                        st.download_button(
                            label="Download",
                            data=content,
                            file_name=notes_filename,
                            mime="text/markdown",
                            key=f"download_{note_id}"
                        )
        else:
            st.info("Upload a document to generate notes")
    
    # History tab
    with tab4:
        st.subheader("Study History")
        if st.session_state.pdf_id:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Important Concepts")
                cursor = conn.cursor()
                cursor.execute("SELECT query, response FROM queries WHERE pdf_id = ? AND is_important = 1 ORDER BY timestamp DESC", 
                              (st.session_state.pdf_id,))
                important_queries = cursor.fetchall()
                
                if important_queries:
                    for q, r in important_queries:
                        with st.expander(f"Q: {q}"):
                            st.write(f"**A:** {r}")
                else:
                    st.info("No important concepts marked yet")
            
            with col2:
                st.markdown("### Recent Questions")
                cursor.execute("SELECT query, response, timestamp FROM queries WHERE pdf_id = ? ORDER BY timestamp DESC LIMIT 10", 
                              (st.session_state.pdf_id,))
                history = cursor.fetchall()
                
                if history:
                    for q, r, t in history:
                        with st.expander(f"Q: {q}"):
                            st.write(f"**A:** {r}")
                            st.write(f"**Time:** {t}")
                else:
                    st.info("No query history yet")
        else:
            st.info("Upload a document to see your study history")

if __name__ == "__main__":
    main()
