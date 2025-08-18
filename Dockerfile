# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies and Redis server
RUN apt-get update && \
    apt-get install -y postgresql-client redis redis-tools && \
    pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Flask port
EXPOSE 8001

# Set environment variables for Flask
ENV FLASK_APP=main.py
ENV FLASK_RUN_PORT=8001

# Add Redis configuration
COPY redis.conf /etc/redis/redis.conf

# Copy entrypoint script and set permissions
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose Redis port
EXPOSE 6379

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

