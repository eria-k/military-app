# Placeholder — will add model download & extraction logic here later
import os
os.system("gunicorn app:app --bind 0.0.0.0:$PORT --worker-class gthread --workers 1 --threads 8 --timeout 600")
