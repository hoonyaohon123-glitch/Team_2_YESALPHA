#!/bin/bash
echo "Starting Docker Environment..."
docker-compose up -d --build

# To run a specific pipeline file, uncomment the line below:
# docker-compose exec pipeline python src/main.py

echo "Environment is up and running!"
