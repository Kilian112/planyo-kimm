#!/bin/bash

# Start Redis in the background
redis-server /etc/redis/redis.conf &
echo "Starting Redis..."

# Wait for Redis to be ready
until redis-cli -h 127.0.0.1 -p 6379 -a "$REDIS_PASSWORD" ping | grep -q PONG; do
    echo "Redis is unavailable - sleeping"
    sleep 1
done

echo "Redis is up - starting Gunicorn servers"

# Start both Gunicorn instances
gunicorn -b 0.0.0.0:8001 -w 4 -k sync --timeout 120 --preload main:app &
gunicorn -b 0.0.0.0:4200 -w 4 -k sync --timeout 120 --preload main:app