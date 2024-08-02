# Use a base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Copy the application code
COPY . .

# Set environment variable
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"]
