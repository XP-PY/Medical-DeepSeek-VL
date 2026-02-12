from dotenv import load_dotenv
import os

load_dotenv()
MODEL_DIR = os.getenv('MODEL_DIR', '/default/path')
DATA_DIR = os.getenv('DATA_DIR', '/default/path')
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv('CUDA_VISIBLE_DEVICES', '0')
os.environ["HF_ENDPOINT"] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')