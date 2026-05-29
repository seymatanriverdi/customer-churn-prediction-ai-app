import { useState } from "react";

function App() {
  const [form, setForm] = useState({
    gender_Male: 1,
    SeniorCitizen: 0,
    Partner_Yes: 1,
    Dependents_Yes: 0,
    tenure: 24,
    MonthlyCharges: 70,
    TotalCharges: 1500,
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: Number(e.target.value),
    });
  };

  const getRiskLabel = (probability) => {
    if (probability < 0.3) return "Low Risk";
    if (probability < 0.6) return "Medium Risk";
    return "High Risk";
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(form),
    });

    const data = await response.json();
    setResult(data);
  };

  return (
    <div style={{ padding: "40px", fontFamily: "Arial", maxWidth: "650px" }}>
      <h1>Customer Churn Prediction</h1>
      <p>Enter customer information to estimate churn risk.</p>

      <form onSubmit={handleSubmit}>
        {Object.keys(form).map((key) => (
          <div key={key} style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", marginBottom: "4px" }}>
              {key}
            </label>
            <input
              name={key}
              type="number"
              value={form[key]}
              onChange={handleChange}
              style={{
                width: "100%",
                padding: "8px",
                border: "1px solid #ccc",
                borderRadius: "6px",
              }}
            />
          </div>
        ))}

        <button
          type="submit"
          style={{
            padding: "10px 16px",
            borderRadius: "6px",
            border: "none",
            cursor: "pointer",
          }}
        >
          Predict Churn Risk
        </button>
      </form>

      {result && (
        <div
          style={{
            marginTop: "24px",
            padding: "16px",
            border: "1px solid #ddd",
            borderRadius: "8px",
          }}
        >
          <h2>Prediction Result</h2>
          <p>
            <strong>Prediction:</strong>{" "}
            {result.prediction === 1 ? "Churn" : "No Churn"}
          </p>
          <p>
            <strong>Churn Probability:</strong>{" "}
            {(result.churn_probability * 100).toFixed(2)}%
          </p>
          <p>
            <strong>Risk Level:</strong>{" "}
            {getRiskLabel(result.churn_probability)}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;