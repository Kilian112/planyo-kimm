
bind = "0.0.0.0:8001"
workers = 4  # Adjust based on your system's resources
worker_class = "sync"  # Use "sync" or "gevent"
timeout = 120
preload_app = True


