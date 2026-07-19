FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for better Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project into the container
COPY . .

# Expose Gradio default port
EXPOSE 7860

# Run the Gradio app
CMD ["python", "app/main.py"]