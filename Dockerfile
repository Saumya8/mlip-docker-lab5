# syntax=docker/dockerfile:1

FROM python:3.9-slim

# TODO [1]: Set WORKDIR to /app
WORKDIR /app

# TODO [2.1]: Copy the requirements.txt file to the WORKDIR
COPY requirements.txt /app/

# Deliverable 3: Include trained model
COPY iris_model.pkl /app/

# TODO [2.2]: Install the requirements
RUN pip install --no-cache-dir -r requirements.txt

# TODO [3]: Copy the server to the WORKDIR
COPY . /app/

# TODO [4]: Set the command to run the app
CMD ["python", "server.py"]
#CMD [...]