#Use the aws lambda image as base image
FROM public.ecr.aws/lambda/python:3.11.9

#Install build-essential compiler and tools
RUN microdnf update -y && microdnf install -y c++ make

# Set the working directory to /app
WORKDIR /app

#Copy requirements.txt to the container
COPY requirements.txt ${LAMBDA_TASK_ROOT}

#Install the dependencies
RUN pip install -r requirements.txt

#Copy the application code to the container
COPY src/travel_agent.py ./src ${LAMBDA_TASK_ROOT}

#Copy the api_key.py file to the container
COPY config ./config ${LAMBDA_TASK_ROOT}

#Set the permissions to the travel_agent.py file
RUN chmod +x /src/travel_agent.py

#Set the CMD to your handler (could also be done as a
# parameter override outside of the Dockerfile)
CMD ["travel_agent.lambda_handler"]

# Metadata
LABEL authors="diegodemiranda"



