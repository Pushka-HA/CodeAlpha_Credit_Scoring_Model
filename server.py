import http.server
import json
import joblib
import numpy as np

# Load trained model
model = joblib.load("credit_model.joblib")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Credit Scoring Model - CodeAlpha</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 30px; display: flex; justify-content: center; }
        .card { background: #1e293b; border-radius: 12px; padding: 25px; width: 440px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
        h2 { text-align: center; color: #38bdf8; margin-top: 0; }
        p.sub { text-align: center; color: #94a3b8; font-size: 13px; margin-bottom: 20px; }
        label { display: block; margin-top: 14px; font-size: 14px; color: #cbd5e1; }
        input { width: 100%; padding: 10px; margin-top: 5px; border-radius: 6px; border: 1px solid #334155; background: #0f172a; color: #fff; box-sizing: border-box; }
        button { width: 100%; margin-top: 20px; padding: 12px; background: #2563eb; color: #fff; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; font-size: 15px; }
        button:hover { background: #1d4ed8; }
        #result { margin-top: 20px; padding: 16px; border-radius: 6px; display: none; }
        .approved { background: rgba(34, 197, 94, 0.2); border: 1px solid #22c55e; color: #4ade80; }
        .rejected { background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #f87171; }
        .score-val { font-size: 22px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="card">
        <h2>💳 Credit Scoring Model</h2>
        <p class="sub">CodeAlpha Machine Learning Internship - Task 1</p>
        <form id="form">
            <label>Monthly/Annual Income (₹):</label>
            <input type="number" id="income" value="60000" required>
            
            <label>Total Outstanding Debt (₹):</label>
            <input type="number" id="debt" value="15000" required>
            
            <label>Credit Utilization (0.0 to 1.5):</label>
            <input type="number" step="0.01" id="util" value="0.25" required>
            
            <label>Past Delinquencies:</label>
            <input type="number" id="delinq" value="0" required>
            
            <button type="submit">Evaluate Creditworthiness</button>
        </form>
        <div id="result"></div>
    </div>

    <script>
        document.getElementById("form").addEventListener("submit", async (e) => {
            e.preventDefault();
            const payload = {
                income: parseFloat(document.getElementById("income").value),
                total_debt: parseFloat(document.getElementById("debt").value),
                credit_utilization: parseFloat(document.getElementById("util").value),
                past_delinquencies: parseInt(document.getElementById("delinq").value)
            };

            const res = await fetch("/score", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            const resDiv = document.getElementById("result");
            resDiv.style.display = "block";
            resDiv.className = data.decision === "APPROVED" ? "approved" : "rejected";
            resDiv.innerHTML = `
                <div class="score-val">Decision: ${data.decision}</div>
                <p><b>Estimated Credit Score:</b> ${data.score} / 900</p>
                <p><b>Default Probability:</b> ${data.default_prob}%</p>
                <p><b>Risk Tier:</b> ${data.risk_tier}</p>
            `;
        });
    </script>
</body>
</html>
"""

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path == "/score":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))

            dti = data["total_debt"] / data["income"]
            features = [[
                data["income"],
                data["total_debt"],
                dti,
                data["credit_utilization"],
                data["past_delinquencies"]
            ]]

            prob = float(model.predict_proba(features)[0][1])
            score = int((1 - prob) * 600 + 300)
            decision = "APPROVED" if prob < 0.40 else "REJECTED"
            risk_tier = "HIGH" if prob >= 0.6 else ("MEDIUM" if prob >= 0.4 else "LOW")

            response = {
                "score": score,
                "default_prob": round(prob * 100, 2),
                "decision": decision,
                "risk_tier": risk_tier
            }

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))

if __name__ == "__main__":
    server = http.server.HTTPServer(("0.0.0.0", 8080), RequestHandler)
    print("Serving Credit Scoring UI on http://localhost:8080 ...")
    server.serve_forever()
