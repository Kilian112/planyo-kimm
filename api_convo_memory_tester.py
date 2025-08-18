from flask import Flask, request, jsonify, session
import redis
import json
import secrets

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Secure session key

# Configure Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

@app.route('/')
def home():
    """Default route to confirm the server is running."""
    return "Memory Test API is running. Use the `/memory-test` endpoint to interact with it."

@app.route('/memory-test', methods=['POST'])
def memory_test():
    """Test conversational memory with Redis."""
    try:
        # Parse incoming JSON request
        data = request.json
        query_text = data.get('query')
        if not query_text:
            return jsonify({"error": "Query text is required"}), 400

        # Manage session ID for conversation memory
        if 'session_id' not in session:
            session['session_id'] = secrets.token_hex(16)
        session_id = session['session_id']
        conversation_key = f"conversation:{session_id}"

        # Retrieve conversation history from Redis
        history_json = redis_client.get(conversation_key)
        history = json.loads(history_json) if history_json else []

        # Append user query to history
        history.append({"role": "user", "content": query_text})

        # Generate a simulated AI response
        ai_response = f"Echoing: {query_text}"
        history.append({"role": "assistant", "content": ai_response})

        # Save updated history back to Redis
        redis_client.set(conversation_key, json.dumps(history))

        # Return the updated history and AI response
        return jsonify({"history": history, "AI Response": ai_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app with debugging and bind to all network interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)
