FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the entire current directory into the container
COPY . .

# Run the training command during the build phase
RUN python train.py

# Command to run when the container starts
CMD ["python", "test.py"]