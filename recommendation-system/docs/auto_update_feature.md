# Auto-Update Implementation Documentation

## Overview

This document describes the implementation of the auto-update feature for the recommendation system. The feature allows the recommendation models to be automatically updated when new location data is imported into MongoDB.

## Components

### 1. MongoDB Change Stream Monitor (mongodb_monitor.py)

This service monitors the MongoDB 'places' collection for changes using Change Streams. When changes are detected, it triggers a model update.

Key features:
- Uses MongoDB Change Streams API to watch for document insertions, updates, and deletions
- Implements debounce mechanism to prevent frequent updates
- Provides methods to start, stop, and check status of the monitor
- Includes a force update method for manual triggering

### 2. Data Importer Service (data_importer.py)

This service provides functionality to import data into MongoDB from CSV or JSON files and automatically trigger model updates.

Key features:
- Supports both CSV and JSON file formats
- Handles data validation and transformation
- Stores a backup of the imported data in the data/raw directory
- Triggers model updates after successful import

### 3. API Endpoints (main.py)

The following API endpoints were added to control the auto-update feature:

- `POST /admin/monitor` - Control the MongoDB monitor
  - `?action=start` - Start the monitor
  - `?action=stop` - Stop the monitor
  - `?action=status` - Check monitor status
  - `?action=force_update` - Force immediate update

- `POST /admin/import` - Import data from CSV or JSON files
  - Requires a file upload (multipart/form-data)
  - Optional 'collection' parameter to specify target collection

- `POST /webhook/data-update` - Webhook for external systems to trigger updates
  - Accepts JSON payload with 'type' and optional 'data_url' fields

### 4. Model Reinitialization Function (main.py)

The `reinitialize_models` function refreshes the recommendation models with the latest data:

- Reloads location details from the updated data files
- Retrains the Content-Based Filtering model
- Updates realtime integration components
- Clears the recommendation cache to ensure fresh recommendations

## Usage

1. **Automatic Updates**: The system will automatically detect changes in MongoDB and update models.

2. **Manual Control**:
   - Start/stop monitoring: `POST /admin/monitor?action=start|stop`
   - Force immediate update: `POST /admin/monitor?action=force_update`

3. **Data Import**:
   ```
   # Import CSV file
   curl -X POST -F "file=@new_locations.csv" http://localhost:5000/admin/import
   
   # Import JSON file
   curl -X POST -F "file=@new_locations.json" http://localhost:5000/admin/import
   ```

4. **Webhook Integration**:
   ```
   curl -X POST -H "Content-Type: application/json" \
     -d '{"type": "location_update", "data_url": "https://example.com/data.csv"}' \
     http://localhost:5000/webhook/data-update
   ```

## Testing

A test script is provided in `test/test_auto_update.py` that demonstrates:
1. Adding a new location directly to MongoDB and observing the auto-update
2. Using the data import API to bulk import locations

Run the test with:
```
python test/test_auto_update.py
```
