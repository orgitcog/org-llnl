import os
from ossp.ossp_server import create_app


if __name__ == "__main__":
    app = create_app()
    # Disable debug in production
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
