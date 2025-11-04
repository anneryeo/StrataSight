# StrataSight CRUD Operations Documentation

## Overview
StrataSight now implements a complete CRUD (Create, Read, Update, Delete) system for managing prediction history.

## Database
- **Type**: JSON-based file storage
- **Location**: `data/predictions_history.json`
- **Format**: Array of prediction records

## CRUD Operations Implementation

### 1. CREATE - Save Predictions
**Location**: Generate Predictions tab → Save This Prediction expander

**Function**: `create_prediction_record()`

**Features**:
- Save current prediction results to database
- Add optional notes/commentary
- Auto-generates unique ID and timestamp
- Stores: ticker, forecast days, current price, LSTM/Prophet predictions, notes

**User Flow**:
1. Generate predictions
2. Click "Save This Prediction" expander
3. Add notes (optional)
4. Click "Save Prediction to History"
5. Confirmation message displays with record ID

---

### 2. READ - View All Predictions
**Location**: Prediction History (CRUD) tab

**Function**: `read_predictions()`, `load_predictions_db()`

**Features**:
- Display all saved predictions in a table
- Shows: ID, timestamp, ticker, forecast days, prices, predictions, notes
- Counter showing total saved predictions
- Full DataFrame view with all columns

**User Flow**:
1. Navigate to "Prediction History (CRUD)" tab
2. View complete table of all saved predictions
3. Browse historical data

---

### 3. UPDATE - Edit Prediction Notes
**Location**: Prediction History tab → Update section (left column)

**Function**: `update_prediction_notes()`

**Features**:
- Select prediction by ID
- View current notes
- Edit notes in text area
- Save updated notes with timestamp
- Adds `last_updated` field to record

**User Flow**:
1. Navigate to Prediction History tab
2. Select prediction ID from dropdown (left column)
3. View current notes
4. Enter new notes
5. Click "Update Notes"
6. Confirmation message and page refresh

---

### 4. DELETE - Remove Predictions
**Location**: Prediction History tab → Delete section (right column)

**Function**: `delete_prediction()`

**Features**:
- Select prediction by ID
- Preview prediction details before deletion
- Warning message with ticker and date
- Permanent removal from database

**User Flow**:
1. Navigate to Prediction History tab
2. Select prediction ID from dropdown (right column)
3. Review warning message
4. Click "Delete Prediction"
5. Confirmation message and page refresh

---

## Additional Features

### Bulk Operations
**Location**: Prediction History tab → Bulk Operations section

**Export All Predictions**:
- Download complete database as JSON file
- Timestamped filename for backup purposes

**Delete All Predictions**:
- Clear entire database
- Warning: permanent action with no confirmation dialog

---

## Database Schema

### Prediction Record Structure
```json
{
  "id": "1",
  "timestamp": "2025-11-04 14:30:45",
  "ticker": "TSLA",
  "forecast_days": 30,
  "current_price": 245.67,
  "lstm_prediction": 267.89,
  "prophet_prediction": 252.34,
  "notes": "Strong upward trend expected",
  "last_updated": "2025-11-04 15:20:10"  // Added on UPDATE
}
```

### Fields:
- **id**: String, unique identifier (sequential)
- **timestamp**: String, creation datetime (YYYY-MM-DD HH:MM:SS)
- **ticker**: String, stock ticker symbol
- **forecast_days**: Integer, prediction horizon
- **current_price**: Float, stock price at prediction time
- **lstm_prediction**: Float or null, LSTM model forecast
- **prophet_prediction**: Float or null, Prophet model forecast
- **notes**: String, user commentary (default: empty)
- **last_updated**: String, last modification datetime (optional)

---

## Technical Implementation

### File Operations
- **Load**: `load_predictions_db()` reads JSON file
- **Save**: `save_predictions_db(predictions)` writes JSON file
- **Auto-create**: Creates `data/` directory if not exists
- **Error Handling**: Returns empty array if file doesn't exist

### Streamlit Integration
- **Tabs**: Two-tab interface (Generate | History)
- **Rerun**: Uses `st.rerun()` after UPDATE/DELETE for immediate UI refresh
- **State Management**: Loads fresh data on each page interaction
- **User Feedback**: Success/error messages for all operations

### Data Persistence
- JSON format for human-readable storage
- No external database dependencies
- Portable across systems (included in project folder)
- Version control friendly (text-based format)

---

## Use Cases for Academic Documentation

### Why This is a CRUD Application:
1. **CREATE**: Users can save new prediction records with notes
2. **READ**: Complete prediction history viewable in table format
3. **UPDATE**: Edit notes on existing predictions without deleting
4. **DELETE**: Remove individual or all predictions from database

### Business Logic:
- **Prediction Management**: Track historical forecasts for analysis
- **Note-taking**: Document reasoning or market conditions
- **Comparison**: Review past predictions vs actual outcomes
- **Audit Trail**: Timestamps track when predictions were made

### CRUD Justification:
StrataSight combines **ML inference** (generate predictions) with **data management** (CRUD operations on prediction history), making it a hybrid ML application with database functionality.

---

## Files Modified
- `src/stratasight.py`: Added CRUD functions and history tab
- `data/predictions_history.json`: Auto-created database file

## Dependencies
- No new dependencies required (uses built-in `json` module)
