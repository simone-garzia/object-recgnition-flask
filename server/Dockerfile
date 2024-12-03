# server/Dockerfile
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy files
COPY . /app/

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Run the server
CMD ["python", "app.py"]
