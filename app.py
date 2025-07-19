from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and expected column structure
model = joblib.load('loan_model.pkl')
model_columns = joblib.load('model_columns.pkl')  # must be saved in notebook

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Collect inputs from form
            input_dict = {
                'Gender': request.form['Gender'],
                'Married': request.form['Married'],
                'Dependents': request.form['Dependents'],
                'Education': request.form['Education'],
                'Self_Employed': request.form['Self_Employed'],
                'ApplicantIncome': float(request.form['ApplicantIncome']),
                'CoapplicantIncome': float(request.form['CoapplicantIncome']),
                'LoanAmount': float(request.form['LoanAmount']),
                'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
                'Credit_History': float(request.form['Credit_History']),
                'Property_Area': request.form['Property_Area']
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([input_dict])

            # One-hot encode
            input_encoded = pd.get_dummies(input_df)

            # Align with training columns
            input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

            # Predict
            prediction = model.predict(input_aligned)
            output = "✅ Loan Approved" if prediction[0] == 1 else "❌ Loan Rejected"
            return render_template('index.html', prediction_text=output)

        except Exception as e:
            return render_template('index.html', prediction_text="Error: " + str(e))

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

