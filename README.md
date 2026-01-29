# YouTube-Creator-Tool
 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index 83464e97e6557e3cf1c66b830aba5207680adb51..a87514ff52b918761183d5806b4d7ccb101fe9fc 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,32 @@
-# YouTube-Creator-Tool
\ No newline at end of file
+# YouTube Creator Tool
+
+A Streamlit dashboard that analyzes YouTube channel performance and predicts next video topics using recent title/tag trends. It pulls data from the YouTube Data API and summarizes upload frequency, engagement metrics, and view trends.
+
+## Features
+- Authenticate with the YouTube Data API.
+- Fetch channel uploads and key video metadata (views, likes, comments).
+- Analyze upload frequency and engagement averages.
+- Visualize views over time and posting cadence.
+- Predict next video topics based on recency-weighted keyword trends.
+
+## Getting Started
+
+### 1. Install dependencies
+```bash
+pip install -r requirements.txt
+```
+
+### 2. Run the dashboard
+```bash
+streamlit run app.py
+```
+
+### 3. Use the app
+- Enter your **YouTube Data API key**.
+- Provide a **channel ID**.
+- Choose how many recent videos to analyze.
+- Click **Fetch Insights** to generate charts and topic predictions.
+
+## Notes
+- The “Next Video Topic Prediction” feature uses titles and tags from recent uploads to surface high-momentum keywords and suggested topics.
+- If likes or comments are disabled, the dashboard will display available metrics and show the ratio as N/A.
 
EOF
)
