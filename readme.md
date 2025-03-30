# Course Study Agent API (PDFs)

This project implements a RAG-based chat API with an agentic flow to help users study course materials. It uses FastAPI, LangGraph, LangChain, and FAISS for vector storage.
LLM-> gpt-4o-mini

## Goals

1.  **Ingest Sample Course Data:(PDFs)**
    * Ingests PDF course materials.
    * Converts content to embeddings and stores them in a FAISS vector database.
2.  **Build a RAG-based Chat API:**
    * Uses FastAPI to provide a RESTful endpoint.
    * Retrieves relevant content chunks and uses an LLM to generate answers.
3.  **Create an Agentic Flow (Plan → Act → Observe → Reflect):**
    * Simulates goal-driven behavior.
    * Uses LangGraph for the agentic workflow.

## Setup Instructions

*Please create a virtual environment and install all requiremnts.*

**I have already created FAISS INDEX which is stored in folder faiss_indexes with my course material(NCERT BOOK of SCience for class 10th.)
Feel free to use your own pdfs.
Add your pdfs in my_pdfs directory.
If you want to use your own pdfs delete the existing faiss_index folder.
New faiss index will be created with same name(faiss_indexes).**


1.  **Create a Virtual Environment:**

    ```
    python -m venv venv
    
    ```

2.  **Install Dependencies:**

    ```
    pip install -r requirements.txt
    ```


3.  **Set Up OpenAI API Key:**

    * Create a `.env` file in the project's root directory.
    * Add your OpenAI API key to the `.env` file:

        ```
        OPENAI_API_KEY=your_openai_api_key
        ```

4.  **Prepare Course Materials:**

    * Create a directory named `my_pdfs` in the project's root directory.
    * Place your PDF course materials inside the `my_pdfs` directory.

4.  **Run the FastAPI Application:**

    ```
    uvicorn main:app --reload
    ```

    Replace `main:app` with the filename and FastAPI app instance if they are different.

    **example**
    ```
    uvicorn agentic_full_v1:app --reload
    ```

## Usage Instructions

### Chat API Endpoint

* **Endpoint:** `POST /query/`
* **Request Body:**

    ```
    {
        "query": "Your question or study objective here"
    }
    ```

* **Response:**

    ```
    {
        "response": "The agent's response based on your query"
    }
    ```

### Log Retrieval Endpoint

* **Endpoint:** `GET /logs_json/`
* **Purpose:** Retrieves the application's log file as JSON.


* **Response:**

    ```
    {
        "log_lines": [
            "Log line 1",
            "Log line 2",
            "..."
        ]
    }
    ```

### Clear Logs Endpoint

* **Endpoint:** `POST /clear_logs/`
* **Purpose:** Clears the application's log file.


* **Response:**

    ```
    {
        "message": "Logs cleared successfully"
    }
    ```

## Notes

* Ensure your OpenAI API key is valid.
* Place your PDF files in the `my_pdfs` directory before running the application.
* The FAISS vector store is saved locally in the `faiss_indexes` directory.
* Adjust the `log_file` variable in `agentic_full_v1.py` if you want to change the log file's name or location.
* The application uses `gpt-4o-mini` by default. Change the model in `agentic_full_v1.py` if needed.