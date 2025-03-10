from flask import Flask, request, render_template
import joblib

# To Load the model and encoders
clf = joblib.load('tic_tac_toe_random_forest_model.pkl')
encoders = joblib.load('feature_encoders.pkl')
le_target = joblib.load('target_encoder.pkl')

columns = ["top-left", "top-middle", "top-right", 
           "middle-left", "middle-middle", "middle-right", 
           "bottom-left", "bottom-middle", "bottom-right"]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        your_board = [
            request.form['top-left'], request.form['top-middle'], request.form['top-right'],
            request.form['middle-left'], request.form['middle-middle'], request.form['middle-right'],
            request.form['bottom-left'], request.form['bottom-middle'], request.form['bottom-right']
        ]

        encoded_board = []
        for i, col in enumerate(columns):
            le = encoders[col]
            encoded_board.append(le.transform([your_board[i]])[0])
        encoded_board = [encoded_board]

        prediction = clf.predict(encoded_board)[0]
        prediction = "x has won!" if prediction == 1 else "x has not won."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)