from flask import Flask
from config import Config

from extensions import init_openai
from blueprints.retriever import retriever_bp


from flask_cors import CORS


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Allow all origins with comprehensive settings
    CORS(app, 
         supports_credentials=True, 
         origins="*",
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
         allow_headers=['Content-Type', 'Authorization', 'X-Requested-With', 'Accept', 'Origin'],
         expose_headers=['Content-Type', 'Authorization']
    )

    # Initialize extensions
    init_openai(app)

    # Register blueprints
    app.register_blueprint(retriever_bp, url_prefix="/retriever")

    
    return app