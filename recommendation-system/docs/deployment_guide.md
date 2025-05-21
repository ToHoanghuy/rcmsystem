# Recommendation System Deployment Guide

This guide provides instructions for deploying the travel recommendation system to a production environment.

## Prerequisites

- Linux-based server (Ubuntu 20.04+ recommended)
- Python 3.8+ installed
- MongoDB 4.4+ installed (optional but recommended)
- Nginx or another web server for reverse proxy
- Domain name (optional)

## 1. System Setup

### 1.1 Create a Dedicated User

```bash
sudo adduser recommender
sudo usermod -aG sudo recommender
su - recommender
```

### 1.2 Install Required Software

```bash
# Update system packages
sudo apt update
sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools python3-venv mongodb nginx supervisor git
```

### 1.3 Install MongoDB (if not already installed)

```bash
# Import MongoDB public key
wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | sudo apt-key add -

# Create list file for MongoDB
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list

# Update the package list
sudo apt-get update

# Install MongoDB
sudo apt-get install -y mongodb-org

# Start MongoDB service
sudo systemctl start mongod

# Enable MongoDB to start on system boot
sudo systemctl enable mongod
```

## 2. Application Setup

### 2.1 Clone the Repository

```bash
# Clone the repository
git clone https://github.com/your-username/travel-recommendation-system.git
cd travel-recommendation-system/python/recommendation-system
```

### 2.2 Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Add MongoDB packages
pip install pymongo
```

### 2.3 Configure Environment Variables

Create a `.env` file in the recommendation-system directory:

```bash
# Create .env file
cat > .env << EOL
MONGODB_CONNECTION_STRING=mongodb://localhost:27017/
FLASK_ENV=production
FLASK_SECRET_KEY=$(openssl rand -hex 24)
CACHE_SIZE=2000
CACHE_TTL=3600
LOG_LEVEL=INFO
EOL
```

### 2.4 Set Up Configuration

Create or modify the config file to use the environment variables:

```python
# config/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Base directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    
    # MongoDB configuration
    MONGODB_CONNECTION_STRING = os.getenv('MONGODB_CONNECTION_STRING', 'mongodb://localhost:27017/')
    
    # Cache configuration
    CACHE_SIZE = int(os.getenv('CACHE_SIZE', 2000))
    CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))
    
    # Flask configuration
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'development-key')
    DEBUG = os.getenv('FLASK_ENV', 'production') != 'production'
    
    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Create directories if they don't exist
    @classmethod
    def create_directories(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
```

## 3. Set Up Supervisor

Create a supervisor configuration to keep the application running:

```bash
# Create supervisor config
sudo vim /etc/supervisor/conf.d/recommendation-system.conf
```

Add the following content:

```ini
[program:recommendation-system]
directory=/home/recommender/travel-recommendation-system/python/recommendation-system
command=/home/recommender/travel-recommendation-system/python/recommendation-system/venv/bin/python main.py
user=recommender
autostart=true
autorestart=true
stderr_logfile=/var/log/recommendation-system/err.log
stdout_logfile=/var/log/recommendation-system/out.log
environment=PYTHONPATH="/home/recommender/travel-recommendation-system/python/recommendation-system"
```

Create the log directory:

```bash
# Create log directory
sudo mkdir -p /var/log/recommendation-system
sudo chown -R recommender:recommender /var/log/recommendation-system
```

Update supervisor:

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start recommendation-system
```

## 4. Set Up Nginx as Reverse Proxy

Install Nginx if not already installed:

```bash
sudo apt install -y nginx
```

Create an Nginx configuration file:

```bash
# Create Nginx config
sudo vim /etc/nginx/sites-available/recommendation-system
```

Add the following content:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain or server IP

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable the configuration:

```bash
sudo ln -s /etc/nginx/sites-available/recommendation-system /etc/nginx/sites-enabled/
sudo nginx -t  # Test the configuration
sudo systemctl restart nginx
```

## 5. SSL/TLS Configuration (Optional)

Install Certbot for SSL/TLS certificates:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 6. Monitoring Setup

### 6.1 Install Monitoring Tools

```bash
# Install Prometheus Node Exporter
sudo apt install -y prometheus-node-exporter

# Start and enable the service
sudo systemctl start prometheus-node-exporter
sudo systemctl enable prometheus-node-exporter
```

### 6.2 Set Up Application Monitoring

Add Python monitoring with Prometheus client:

```bash
pip install prometheus-client
```

Add to your main.py file:

```python
from prometheus_client import start_http_server, Counter, Histogram
import time

# Set up metrics
REQUESTS = Counter('recommendation_requests_total', 'Total number of recommendation requests', ['endpoint'])
RESPONSE_TIME = Histogram('recommendation_response_time_seconds', 'Response time in seconds', ['endpoint'])
RECOMMENDATIONS_SERVED = Counter('recommendations_served_total', 'Total number of recommendations served')
EVENTS_PROCESSED = Counter('events_processed_total', 'Total number of events processed', ['event_type'])

# Start Prometheus metrics server on another port
start_http_server(9090)
```

## 7. Database Backup

Set up a daily backup of MongoDB:

```bash
# Create backup script
cat > /home/recommender/backup-mongodb.sh << EOL
#!/bin/bash
BACKUP_DIR="/home/recommender/backups/mongodb"
DATE=\$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p \$BACKUP_DIR
mongodump --out=\$BACKUP_DIR/\$DATE
find \$BACKUP_DIR -type d -mtime +7 -exec rm -rf {} \;
EOL

# Make the script executable
chmod +x /home/recommender/backup-mongodb.sh

# Add to crontab
(crontab -l 2>/dev/null; echo "0 3 * * * /home/recommender/backup-mongodb.sh") | crontab -
```

## 8. Start the Application

Start the recommendation system:

```bash
# Using supervisor
sudo supervisorctl start recommendation-system

# Check status
sudo supervisorctl status recommendation-system
```

## 9. Testing the Deployment

Test that the deployment is working correctly:

```bash
# Test the API
curl http://your-domain.com/api/recommend?user_id=1&case=hybrid

# Check the WebSocket test client
# Open in browser: http://your-domain.com/websocket-test
```

## 10. Updating the Application

To update the application:

```bash
# Pull the latest changes
cd /home/recommender/travel-recommendation-system
git pull

# Activate the virtual environment
cd python/recommendation-system
source venv/bin/activate

# Install any new dependencies
pip install -r requirements.txt

# Restart the application
sudo supervisorctl restart recommendation-system
```

## 11. Troubleshooting

### 11.1 Checking Logs

```bash
# Check application logs
sudo tail -f /var/log/recommendation-system/out.log
sudo tail -f /var/log/recommendation-system/err.log

# Check MongoDB logs
sudo tail -f /var/log/mongodb/mongod.log

# Check Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 11.2 Common Issues

- **MongoDB connection issues**: Check if MongoDB is running with `sudo systemctl status mongod`
- **Application won't start**: Check the error log and ensure permissions are correct
- **WebSocket not working**: Ensure Nginx is configured correctly for WebSocket support
- **High CPU/Memory usage**: Check the system resources and optimize the application configuration

## 12. Security Considerations

- Keep all software up to date
- Use a firewall (e.g., UFW) to restrict access to necessary ports
- Set up fail2ban to prevent brute force attacks
- Use strong passwords for all services
- Regularly audit system logs for suspicious activity

By following this deployment guide, you should now have a fully functional recommendation system running in a production environment, with monitoring, backups, and security measures in place.
