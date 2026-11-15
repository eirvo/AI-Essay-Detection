import pickle
from flask import Flask, request, jsonify

with open('model.bin', 'rb') as f:
    pipe = pickle.load(f)

app = Flask('ai-essay-detector')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.get_json().get('text', '')
    prob = float(pipe.predict_proba([text])[0, 1])
    return jsonify({'ai_probability': round(prob, 4), 'is_ai': prob >= 0.5})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)