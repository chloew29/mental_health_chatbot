1. Clone the repo
```bash
git clone 
cd mental_health_chatbot
```

2. Create & activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Training the Model
```bash
cd training_model
python train_risk_detector_optimized.py
```

5. Running the model on Test Data
```bash
python test_risk_detector.py
```

6. Plotting Results
```bash
python plot_results.py
```
