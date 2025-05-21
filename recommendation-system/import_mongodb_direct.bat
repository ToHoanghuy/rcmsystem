@echo off
echo Running MongoDB import script with fixes...

REM Set environment variables if needed
set MONGODB_URI=mongodb://localhost:27017/travel_recommendations

REM Run the import script for each data type
python -c "import pandas as pd; import os; import sys; from config.config import Config; from database.mongodb import MongoDB; mongo_db = MongoDB(); df = pd.read_csv(os.path.join(Config.RAW_DATA_DIR, 'locations.csv'), on_bad_lines='skip'); print(f'Importing {len(df)} locations...'); records = df.to_dict('records'); result = mongo_db.places.insert_many(records, ordered=False); print(f'Imported {len(result.inserted_ids)} locations')"

python -c "import pandas as pd; import os; import sys; from datetime import datetime; from config.config import Config; from pymongo import MongoClient; client = MongoClient('mongodb://localhost:27017/'); db = client['travel_recommendations']; df = pd.read_csv(os.path.join(Config.RAW_DATA_DIR, 'events.csv'), on_bad_lines='skip'); print(f'Importing {len(df)} events...'); events = []; for _, row in df.iterrows(): event = {'user_id': str(row.get('user_id', '')), 'location_id': str(row.get('product_id', row.get('location_id', ''))), 'event_type': row.get('event_type', 'view'), 'timestamp': datetime.now()}; events.append(event); result = db['events'].insert_many(events, ordered=False); print(f'Imported {len(result.inserted_ids)} events')"

python -c "import pandas as pd; import os; import sys; from datetime import datetime; from config.config import Config; from pymongo import MongoClient; client = MongoClient('mongodb://localhost:27017/'); db = client['travel_recommendations']; df = pd.read_csv(os.path.join(Config.RAW_DATA_DIR, 'dataset.csv'), on_bad_lines='skip'); print(f'Importing {len(df)} ratings...'); ratings = []; for _, row in df.iterrows(): if 'user' in df.columns and 'item' in df.columns and 'rating' in df.columns: rating = {'user_id': str(row['user']), 'location_id': str(row['item']), 'rating': float(row['rating']), 'timestamp': datetime.now()}; ratings.append(rating); result = db['ratings'].insert_many(ratings, ordered=False); print(f'Imported {len(result.inserted_ids)} ratings')"

echo Import process completed!
pause
