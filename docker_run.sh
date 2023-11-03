#!bin/bash
docker build -t digits_docker:v1 -f docker/Dockerfile .
echo "Running Docker image..."
docker run -d -v "$(pwd)/models/":/digits/models/ digits_docker:v1
echo "Completed Run ..."