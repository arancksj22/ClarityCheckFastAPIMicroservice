# 1. Use a lightweight Python base image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements first (for better caching)
COPY requirements.txt .

# 4. Install dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code
COPY . .

# 6. Pre-download the ML model during the build phase
# This prevents the app from trying to download 1GB of data when it first starts up.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 7. Expose the port (Render handles mapping, but this is good practice)
EXPOSE 8000

# 8. Command to run the application
# host 0.0.0.0 is crucial for Docker containers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]