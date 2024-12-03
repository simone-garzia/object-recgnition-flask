# Flask CNN Server

A client-server application that utilizes a Flask-based backend to deploy a Convolutional Neural Network (CNN) model for inference. The project includes a client script to send requests to the server and a backend that handles requests, processes data, and returns results.

### **Server**
- `app.py`: The main Flask application that handles HTTP requests, loads the CNN model, and serves predictions.
- `model.py`: Defines the structure of the CNN model.
- `utils.py`: Contains helper functions for preprocessing input data (e.g., images).
- `model_weights.pth`: Pretrained weights for the CNN model.

### **Client**
- `client.py`: A Python script to interact with the Flask server by sending data for inference and receiving results.

### **Dependencies**
- `requirements.txt`: Lists all the Python libraries required to run the application.

---

## Requirements
The following Python libraries are required:
- `flask`
- `torch`
- `torchvision`
- `Pillow`
- `requests`

---

## Installation and Setup

### **1. Clone the Repository**
   git clone https://github.com/simone-garzia/flask-obj-detection.git
   cd flask_cnn_server
### 2. Set Up a Virtual Environment (Optional)
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
### 3. Install Dependencies
    pip install -r requirements.txt
### 4. Start the Flask Server
    python server/app.py
### 5. Run the Client
    python client/client.py
