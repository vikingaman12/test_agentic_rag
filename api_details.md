# Course Study Agent API Documentation

This document provides detailed information about the API endpoints for the Course Study Agent application.

## Base URL

All API endpoints are relative to the base URL of your application. For example, if your application is running at `http://127.0.0.1:8000`, the full URL for the `/query/` endpoint would be `http://127.0.0.1:8000/query/`.

## Endpoints

### 1. Chat Query Endpoint (`/query/`)

* **Method:** `POST`
* **Description:** This endpoint is used to submit a study query to the agent.
* **Request Body:**
    * **Content-Type:** `application/json`
    * **Schema:**

        ```json
        {
            "query": "string"
        }
        ```

    * **Example Request:**

        ```json
        {
            "query": "Explain the concept of photosynthesis."
        }
        ```

* **Response:**
    * **Content-Type:** `application/json`
    * **Schema:**

        ```json
        {
            "response": "string"
        }
        ```

    * **Example Response:**

        ```json
        {
            "response": "Photosynthesis is the process by which plants convert light energy into chemical energy..."
        }
        ```

    * **Error Responses:**
        * **500 Internal Server Error:** If there's an error during the workflow. The response body will contain an error message.

### 2. Get Logs Endpoint (`/logs_json/`)

* **Method:** `GET`
* **Description:** Retrieves the application's log file content as a JSON array of log lines.
* **Response:**
    * **Content-Type:** `application/json`
    * **Schema:**

        ```json
        {
            "log_lines": ["string", "string", ...]
        }
        ```

    * **Example Response:**

        ```json
        {
            "log_lines": [
                "2024-10-27 10:00:00,000 - INFO - Application started.",
                "2024-10-27 10:01:00,000 - DEBUG - Query received: Explain photosynthesis."
            ]
        }
        ```

    * **Error Responses:**
        * **404 Not Found:** If the log file is not found.
        * **500 Internal Server Error:** If there's an error reading the log file.

### 3. Clear Logs Endpoint (`/clear_logs/`)

* **Method:** `POST`
* **Description:** Clears the application's log file.
* **Response:**
    * **Content-Type:** `application/json`
    * **Schema:**

        ```json
        {
            "message": "string"
        }
        ```

    * **Example Response:**

        ```json
        {
            "message": "Logs cleared successfully"
        }
        ```

    * **Error Responses:**
        * **500 Internal Server Error:** If there's an error clearing the log file.


## Error Handling

The API uses standard HTTP status codes to indicate the success or failure of requests. Error responses will typically include a JSON body with an error message.

## Usage Examples

### Postman Examples

1.  **Chat Query Endpoint (`/query/`)**

    * **Method:** `POST`
    * **URL:** `http://127.0.0.1:8000/query/` (or your application's base URL)
    * **Headers:**
        * `Content-Type: application/json`
    * **Body:**
        * Select "raw" and "JSON" in Postman.
        * Enter the following JSON in the body:

            ```json
            {
                "query": "What are the key concepts of Topic 3?"
            }
            ```

    * Click "Send".
    * The response will be displayed in the "Body" section of Postman.

2.  **Get Logs Endpoint (`/logs_json/`)**

    * **Method:** `GET`
    * **URL:** `http://127.0.0.1:8000/logs_json/`
    * No body or special headers are needed.
    * Click "Send".
    * The response (JSON array of log lines) will be displayed in the "Body" section.

3.  **Clear Logs Endpoint (`/clear_logs/`)**

    * **Method:** `POST`
    * **URL:** `http://127.0.0.1:8000/clear_logs/`
    * No body or special headers are needed.
    * Click "Send".
    * The response (JSON with a message) will be displayed in the "Body" section.
