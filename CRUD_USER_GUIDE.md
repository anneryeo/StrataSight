# StrataSight CRUD User Guide

## 🎯 Quick Start: CRUD Operations

StrataSight now includes a **Prediction History Database** with full CRUD capabilities!

---

## 📋 Interface Overview

The app has **2 main tabs**:
1. **Generate Predictions** - Create forecasts and save them (CREATE)
2. **Prediction History (CRUD)** - Manage saved predictions (READ, UPDATE, DELETE)

---

## CREATE: Save a Prediction

### Step-by-Step:
1. Go to **Generate Predictions** tab
2. Configure settings in sidebar:
   - Select stock ticker (TSLA, GME, AMD, AAPL)
   - Set forecast horizon (7-90 days)
   - Choose models (LSTM, Prophet, or both)
3. Click **Generate Predictions**
4. After results appear, expand **💾 Save This Prediction**
5. Optionally add notes (e.g., "Bull market expected")
6. Click **Save Prediction to History**
7. Confirmation shows: "Prediction saved! (ID: X)"

### What Gets Saved:
- Unique ID
- Timestamp
- Stock ticker
- Forecast horizon
- Current price
- LSTM prediction (if used)
- Prophet prediction (if used)
- Your notes

---

## READ: View All Predictions

### Step-by-Step:
1. Go to **Prediction History (CRUD)** tab
2. View the **All Saved Predictions** table
3. See complete history with all fields

### Table Columns:
| Column | Description |
|--------|-------------|
| id | Unique identifier |
| timestamp | When prediction was created |
| ticker | Stock symbol |
| forecast_days | Prediction horizon |
| current_price | Price at prediction time |
| lstm_prediction | LSTM forecast result |
| prophet_prediction | Prophet forecast result |
| notes | Your commentary |

---

## UPDATE: Edit Prediction Notes

### Step-by-Step:
1. Go to **Prediction History (CRUD)** tab
2. Scroll to **Update or Delete Predictions** section
3. In the **left column** (UPDATE):
   - Select prediction ID from dropdown
   - View current notes
   - Enter new notes in text area
   - Click **Update Notes**
4. Success message appears and table refreshes

### Use Cases:
- Add market context after the fact
- Correct typos in original notes
- Document prediction accuracy after time passes
- Add analysis or lessons learned

---

## DELETE: Remove Predictions

### Step-by-Step:
1. Go to **Prediction History (CRUD)** tab
2. Scroll to **Update or Delete Predictions** section
3. In the **right column** (🗑️ DELETE):
   - Select prediction ID from dropdown
   - Review warning (shows ticker and date)
   - Click **Delete Prediction**
4. Success message appears and table refreshes

### Warning:
Deletion is **permanent** and cannot be undone!

---

## Bulk Operations

### Export All Predictions
**Purpose**: Backup your prediction history

**Steps**:
1. Go to **Prediction History** tab
2. Scroll to **Bulk Operations**
3. Click **Export All Predictions (JSON)**
4. Click **Download JSON** button
5. File saved as: `predictions_export_YYYYMMDD_HHMMSS.json`

**Use Cases**:
- Create backups before major changes
- Share prediction history with team
- Migrate data to another system
- Archive old predictions

---

### Delete All Predictions
**Purpose**: Clear entire database

**Steps**:
1. Go to **Prediction History** tab
2. Scroll to **Bulk Operations**
3. Click **Delete All Predictions**
4. All records deleted immediately

**Warning**:
**NO CONFIRMATION DIALOG** - This action is immediate and permanent!
**Best Practice**: Export before using this feature

---

## Pro Tips

### Organizing Predictions:
- Use consistent note format: `[Date] - [Analysis] - [Expected Outcome]`
- Example: "2025-11-04 - Strong earnings report - Expecting 10% gain"

### Note Templates:
```
Market Sentiment: Bullish/Bearish/Neutral
Reasoning: [Why you made this prediction]
Expected Trend: Up/Down/Sideways
Confidence: High/Medium/Low
```

### Regular Maintenance:
1. Export predictions weekly (backup)
2. Review old predictions monthly
3. Update notes with actual outcomes
4. Delete outdated or incorrect predictions

---

## Example Workflow

### Scenario: Tracking TSLA Stock

**Week 1 - CREATE**:
1. Generate TSLA prediction for 30 days
2. Save with notes: "Q4 earnings coming, bullish outlook"
3. Record ID: 1

**Week 2 - READ**:
1. Check prediction history
2. LSTM predicted $267, currently at $260
3. Prediction tracking well

**Week 3 - UPDATE**:
1. Select prediction ID 1
2. Add notes: "UPDATE: Earnings beat expectations, price target raised"
3. Save changes

**Week 4 - Evaluate**:
1. Final price: $270
2. Add final note: "RESULT: LSTM accurate within 1%, Prophet within 7%"
3. Keep for historical analysis OR delete if no longer needed

---

## Database Location

**File Path**: `data/predictions_history.json`

**Format**: Human-readable JSON
```json
[
  {
    "id": "1",
    "timestamp": "2025-11-04 14:30:45",
    "ticker": "TSLA",
    "forecast_days": 30,
    "current_price": 245.67,
    "lstm_prediction": 267.89,
    "prophet_prediction": 252.34,
    "notes": "Strong upward trend expected"
  }
]
```

**Can be manually edited** if needed (careful with JSON syntax!)

---

## ❓ FAQ

**Q: Can I edit predictions other than notes?**
A: No, only notes can be updated. Price predictions are immutable to maintain data integrity.

**Q: What happens if I delete the JSON file?**
A: The app will create a new empty database. Old predictions are lost unless backed up.

**Q: Is there a limit on saved predictions?**
A: No hard limit, but performance may degrade with thousands of records.

**Q: Can multiple users share the same database?**
A: Currently single-user. The JSON file is in the project directory.

**Q: How do I restore from a backup?**
A: Replace `data/predictions_history.json` with your exported JSON file.

---

## Academic Documentation Notes

### CRUD Evidence for Project Reports:
- **Screenshots needed**:
  1. Save prediction form (CREATE)
  2. Full prediction history table (READ)
  3. Update notes interface (UPDATE)
  4. Delete confirmation (DELETE)

### Database Justification:
- JSON-based NoSQL storage
- File-based persistence
- No external database server required
- Suitable for single-user desktop applications
