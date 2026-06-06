# Complaints Dashboard

A Dash + Plotly recreation of the Chapter 20 complaints dashboard, driven entirely by `26k-consumer-complaints.csv`.

## Run

1. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

2. Place `26k-consumer-complaints.csv` in this directory beside `app.py`.

3. Start the app:

   ```powershell
   python app.py
   ```

4. Open `http://127.0.0.1:8050`.

To load a CSV from another location, set `COMPLAINTS_CSV`:

```powershell
$env:COMPLAINTS_CSV = "D:\26k-consumer-complaints.csv"
python app.py
```

## Interactions

- Date, Source Type, and Open/Closed controls combine with all click filters.
- Click a State, Reason, or Party to filter the dashboard.
- Click the active selection again to remove it.
- `Clear Filters` clears only State, Reason, and Party selections.

The dataset has no `Party` field. The Party chart therefore uses `Company` as a proxy, shows the top eight companies in the current context, and groups the remainder into `Other`.
