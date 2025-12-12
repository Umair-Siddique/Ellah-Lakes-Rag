import os
import sys
from flask import Blueprint, jsonify, request, Response
from openai import OpenAI
from config import Config

retriever_bp = Blueprint("retriever", __name__)