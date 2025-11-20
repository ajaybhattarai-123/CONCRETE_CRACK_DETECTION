# Concrete Crack Detection Web App

This is a Flask web application that uses a pre-trained PyTorch CNN model to detect cracks in concrete images.

## Prerequisites

- Python 3.8+
- A trained PyTorch model file named `best_model.pth` (containing the `state_dict`).

## Installation

1.  **Clone or Download** this project to your local machine.
2.  **Place your model**: Ensure your `best_model.pth` file is in the root directory of this project (same folder as `app.py`).
3.  **Install Dependencies**:
    Open a terminal/command prompt in the project folder and run:
    ```bash
    pip install -r requirements.txt
    ```

## Running the App Locally

1.  **Start the Server**:
    In your terminal, run:
    ```bash
    python app.py
    ```
2.  **Access the App**:
    Open your web browser and go to:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

3.  **Use the App**:
    - Click "Choose File" to select a concrete image.
    - Click "Analyze Image".
    - View the result (Cracked/Not Cracked) and the confidence score.

## Deployment

To "deploy" this locally so others on your Wi-Fi/Network can access it:

1.  Open `app.py`.
2.  Change the last line from:
    ```python
    app.run(debug=True)
    ```
    to:
    ```python
    app.run(host='0.0.0.0', port=5000)
    ```
3.  Run `python app.py` again.
4.  Find your computer's local IP address (run `ipconfig` on Windows or `ifconfig` on Mac/Linux).
5.  Others can access it via `http://YOUR_IP_ADDRESS:5000`.

**Note**: For production deployment (Internet), you would typically use a WSGI server like Gunicorn and host it on a platform like Heroku, AWS, or Render.

## Project Structure

```
/
├── app.py              # Main Flask application
├── best_model.pth      # Your trained model (YOU MUST PROVIDE THIS)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── templates/
    └── index.html      # Frontend HTML template
└── static/
    └── uploads/        # Temporary storage for uploaded images
```
