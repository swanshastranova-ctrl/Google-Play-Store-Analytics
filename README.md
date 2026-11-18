# Google Play Store Analytics — Starter

## Setup
1. Open this folder in VS Code.
2. Create a virtual environment: `python -m venv .venv`
3. Activate on Windows PowerShell: `.venv\Scripts\Activate.ps1`
4. Install: `pip install -r requirements.txt`
5. Run: `streamlit run app.py`
6. Upload the Google Play Store dataset CSV.

This is a starter foundation, not a finished implementation of all six charts. It includes robust initial cleaning, a CSV uploader, and strict Asia/Kolkata chart visibility windows. It deliberately does not fabricate missing country data or monthly install histories.

Expected app columns: App, Category, Rating, Reviews, Size, Installs, Type, Price, Last Updated, Android Ver, Content Rating.
