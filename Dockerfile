# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first (to take advantage of Docker layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


COPY ["predict.py", "predict_sample.py",  "xgb_model_trained.pkl", "./"]

# Set the entry point to run your script
CMD ["python","predict.py"]
