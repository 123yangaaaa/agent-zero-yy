@echo off
echo Starting Agent Zero in Docker...
docker-compose up --build -d
echo Docker container is running!
echo You can access the environment via SSH:
echo Host: localhost
echo Port: 50022
echo Username: root
echo Password: toor
