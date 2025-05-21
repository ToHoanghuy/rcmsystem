# Recommendation System

This project implements a recommendation system designed to provide personalized recommendations based on user-item interactions. The system utilizes machine learning algorithms to analyze user preferences and suggest items accordingly.

## Project Structure

- **data/**: Contains the dataset used for training and evaluating the recommendation system.
  - `dataset.csv`: The dataset file with user-item interactions or relevant features.

- **models/**: Contains the model definition for the recommendation system.
  - `model.py`: Defines the classes and functions for building, training, and evaluating the recommendation algorithm.

- **notebooks/**: Contains Jupyter notebooks for data analysis.
  - `analysis.ipynb`: Used for exploratory data analysis and visualization of the dataset.

- **src/**: Contains the source code for the recommendation system.
  - `main.py`: The entry point of the application, orchestrating the workflow of the recommendation system.
  - `preprocess.py`: Functions for preprocessing the dataset, including data cleaning and transformation.
  - `recommend.py`: Functions for generating recommendations based on the trained model.

- **services/**: Contains services for database and external integrations.
  - `mongodb_monitor.py`: Service to monitor changes in MongoDB and auto-update recommendation models.
  - `data_importer.py`: Service to import data from CSV or JSON files into MongoDB.

- `requirements.txt`: Lists the dependencies required for the project, including libraries for data manipulation, machine learning, and visualization.

## Auto-Update Feature

The system now includes an automatic update mechanism that monitors for changes in MongoDB data and updates the recommendation models when new data is detected:

- **MongoDB Change Stream Monitor**: Listens for changes in MongoDB collections using Change Streams.
- **Data Import API**: Provides endpoints to bulk import data from CSV or JSON files.
- **Webhook Support**: External systems can trigger model updates via webhook.

### Auto-Update API Endpoints

- `POST /admin/monitor?action=start|stop|status|force_update`: Control the MongoDB monitor
- `POST /admin/import`: Upload CSV or JSON files to import into MongoDB
- `POST /webhook/data-update`: Webhook for external systems to trigger model updates

## MongoDB Integration

The recommendation system now supports using MongoDB as a data source instead of CSV files. This allows you to use real data from your MongoDB database for recommendations.

### Prerequisites

- MongoDB Atlas account or a local MongoDB server
- Connection string to your MongoDB database
- Collections for locations, events, and ratings

### Environment Configuration

Configure your MongoDB connection in the `.env` file:

```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database_name?retryWrites=true&w=majority
PLACES_COLLECTION=Location
EVENTS_COLLECTION=events
RATINGS_COLLECTION=ratings
```

### Running with MongoDB Data

Use the provided script to run the recommendation system with MongoDB integration:

Windows:
```
run_mongodb.bat
```

Linux/Mac:
```
./run_mongodb.sh
```

### Testing MongoDB Connection

To test if your MongoDB connection is working correctly, you can run:

```
python test_mongodb_data.py
```

This will attempt to load data from your MongoDB collections and display the results.

### Fallback Mechanism

If MongoDB data is not available or if there are connection issues, the system will automatically fall back to using the CSV files in the `data/raw` directory.

### MongoDB Collections Structure

The system expects the following collections:

1. **Location Collection** (specified by `PLACES_COLLECTION` in config)
   - Fields: `_id`, `name`, `province`, `description`, etc.

2. **Events Collection** (specified by `EVENTS_COLLECTION` in config)
   - Fields: `user_id`, `location_id`, `event_type`, `timestamp`, `data`

3. **Ratings Collection** (specified by `RATINGS_COLLECTION` in config)
   - Fields: `user_id`, `location_id`, `rating`, `timestamp`

### API Endpoints

All existing API endpoints work with MongoDB data without any changes to how they're called.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd recommendation-system
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the recommendation system with CSV data:
```
python main.py
```

To run the recommendation system with MongoDB data:
```
python main_mongodb.py
```

This will load the dataset, preprocess the data, train the model, and generate recommendations based on user input.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd recommendation-system
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the recommendation system, execute the following command:
```
python src/main.py
```

This will load the dataset, preprocess the data, train the model, and generate recommendations based on user input.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.