# Prescripto

1. CREATE AND ACTIVATE A VIRTUAL ENVIRONMENT

Windows (recommended for this project):

Open Command Prompt or VS Code terminal and run:

cd C:\Users\dell\Desktop\prescripto_final
python -m venv venv
.\venv\Scripts\activate

macOS/Linux (if needed):

python3 -m venv venv
source venv/bin/activate

2.INSTALL DEPENDENCIES

With the virtual environment active, run:
pip install -r requirements.txt


3.VERIFY MODEL FILE LOCATION

Ensure the trained model file is present at:
prescripto_final/backend/models/best_model_cer_hybrid_balancedwork.pth


4.START THE BACKEND SERVER

From inside the project folder with the virtual environment activated:

uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

API will run at: http://127.0.0.1:8000


5.START THE FRONTEND INTERFACE

Open another terminal window and run:

cd frontend
python -m http.server 5500

Frontend loads at:

http://127.0.0.1:5500/index.html
