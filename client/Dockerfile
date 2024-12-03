# client/Dockerfile
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy files
COPY . /app/

# Install dependencies
RUN pip install -r requirements.txt

# Run the client (manual trigger or container script can invoke it)
CMD ["python", "client.py"]
