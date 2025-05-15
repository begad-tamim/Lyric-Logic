# Use a base image with Python
FROM python:3.11-slim

# Install PostgreSQL development libraries
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the rest of the application code
COPY ./app /app

# Copy the requirements file
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 80

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
