# Deployment Guide - Pipeline Leak Detection System

This guide covers deploying the Pipeline Leak Detection System in various environments.

## 🚀 Quick Deployment Options

### Option 1: Local Development
```bash
git clone https://github.com/ashish-kj/AnamolyDetector.git
cd AnamolyDetector
pip install -r requirements.txt
python code/live_detection.py
```

### Option 2: Docker Deployment (Recommended for Production)
```bash
git clone https://github.com/ashish-kj/AnamolyDetector.git
cd AnamolyDetector
docker build -t pipeline-detector .
docker run -p 5000:5000 -v /path/to/data:/app/TestData pipeline-detector
```

### Option 3: Cloud Deployment
Deploy to cloud platforms like AWS, Azure, or Google Cloud Platform.

## 🐳 Docker Configuration

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 pipeline && chown -R pipeline:pipeline /app
USER pipeline

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/status || exit 1

# Start application
CMD ["python", "code/live_detection.py"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  pipeline-detector:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./TestData:/app/TestData
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - LOG_LEVEL=INFO
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - pipeline-detector
    restart: unless-stopped
```

## ☁️ Cloud Deployment

### AWS Deployment

#### Using AWS ECS
1. **Build and push Docker image**
   ```bash
   aws ecr create-repository --repository-name pipeline-detector
   docker build -t pipeline-detector .
   docker tag pipeline-detector:latest <account-id>.dkr.ecr.<region>.amazonaws.com/pipeline-detector:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/pipeline-detector:latest
   ```

2. **Create ECS task definition**
   ```json
   {
     "family": "pipeline-detector",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "executionRoleArn": "arn:aws:iam::<account>:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "pipeline-detector",
         "image": "<account-id>.dkr.ecr.<region>.amazonaws.com/pipeline-detector:latest",
         "portMappings": [
           {
             "containerPort": 5000,
             "protocol": "tcp"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/pipeline-detector",
             "awslogs-region": "<region>",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

#### Using AWS Lambda (for batch processing)
```python
import json
import boto3
from code.pipeline_leak_detector import PipelineLeakDetector

def lambda_handler(event, context):
    """AWS Lambda handler for batch leak detection"""
    
    s3_bucket = event['Records'][0]['s3']['bucket']['name']
    s3_key = event['Records'][0]['s3']['object']['key']
    
    # Download file from S3
    s3 = boto3.client('s3')
    local_file = f"/tmp/{s3_key.split('/')[-1]}"
    s3.download_file(s3_bucket, s3_key, local_file)
    
    # Run leak detection
    detector = PipelineLeakDetector()
    results = detector.analyze_pipeline_file(local_file)
    
    # Upload results back to S3
    results_key = f"results/{s3_key.split('/')[-1]}.json"
    s3.put_object(
        Bucket=s3_bucket,
        Key=results_key,
        Body=json.dumps(results, default=str)
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps(f'Processed {s3_key}')
    }
```

### Azure Deployment

#### Using Azure Container Instances
```bash
az group create --name pipeline-detector-rg --location eastus

az container create \
  --resource-group pipeline-detector-rg \
  --name pipeline-detector \
  --image your-registry.azurecr.io/pipeline-detector:latest \
  --cpu 2 \
  --memory 4 \
  --ports 5000 \
  --ip-address public \
  --environment-variables FLASK_ENV=production
```

### Google Cloud Platform

#### Using Cloud Run
```bash
gcloud run deploy pipeline-detector \
  --image gcr.io/PROJECT-ID/pipeline-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --port 5000
```

## 🔧 Production Configuration

### Environment Variables
```bash
# Application settings
FLASK_ENV=production
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here

# Database settings (if using database)
DATABASE_URL=postgresql://user:password@host:port/dbname

# Monitoring settings
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/...
METRICS_ENDPOINT=https://metrics.company.com/api/v1/metrics

# Security settings
ALLOWED_HOSTS=your-domain.com,*.your-domain.com
SSL_REQUIRED=true
```

### Nginx Configuration
```nginx
upstream pipeline_app {
    server 127.0.0.1:5000;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://pipeline_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /socket.io/ {
        proxy_pass http://pipeline_app;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### Systemd Service (Linux)
```ini
[Unit]
Description=Pipeline Leak Detection System
After=network.target

[Service]
Type=simple
User=pipeline
WorkingDirectory=/opt/pipeline-detector
Environment=FLASK_ENV=production
ExecStart=/opt/pipeline-detector/venv/bin/python code/live_detection.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 📊 Monitoring and Logging

### Application Monitoring
```python
# Add to your application
import logging
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
DETECTION_COUNTER = Counter('pipeline_detections_total', 'Total detections')
PROCESSING_TIME = Histogram('pipeline_processing_seconds', 'Processing time')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/pipeline-detector/app.log'),
        logging.StreamHandler()
    ]
)
```

### Health Check Endpoint
```python
@app.route('/health')
def health_check():
    """Health check endpoint for load balancers"""
    try:
        # Basic system checks
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'checks': {
                'database': 'ok',  # if using database
                'disk_space': 'ok',
                'memory': 'ok'
            }
        }
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503
```

## 🔒 Security Configuration

### SSL/TLS Setup
```bash
# Using Let's Encrypt
certbot certonly --webroot -w /var/www/html -d your-domain.com
```

### Firewall Configuration
```bash
# UFW (Ubuntu)
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable

# iptables
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
```

### Application Security
```python
# Add security headers
from flask_talisman import Talisman

Talisman(app, force_https=True)

# Rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
```

## 📈 Scaling Considerations

### Horizontal Scaling
- Use load balancer (nginx, HAProxy, AWS ALB)
- Deploy multiple application instances
- Use Redis for session storage
- Implement database connection pooling

### Vertical Scaling
- Increase CPU and memory allocation
- Optimize detection algorithms
- Use caching for frequently accessed data
- Implement data preprocessing pipelines

### Database Scaling
- Use read replicas for reporting
- Implement database sharding
- Use caching layer (Redis, Memcached)
- Optimize database queries

## 🚨 Disaster Recovery

### Backup Strategy
```bash
# Database backup
pg_dump pipeline_db > backup_$(date +%Y%m%d_%H%M%S).sql

# Application backup
tar -czf app_backup_$(date +%Y%m%d_%H%M%S).tar.gz /opt/pipeline-detector

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump pipeline_db | gzip > $BACKUP_DIR/db_$DATE.sql.gz
find $BACKUP_DIR -name "db_*.sql.gz" -mtime +7 -delete
```

### Recovery Procedures
1. **Application Recovery**
   ```bash
   # Stop services
   systemctl stop pipeline-detector
   
   # Restore application
   tar -xzf app_backup_YYYYMMDD_HHMMSS.tar.gz -C /
   
   # Restore database
   gunzip < db_backup_YYYYMMDD_HHMMSS.sql.gz | psql pipeline_db
   
   # Start services
   systemctl start pipeline-detector
   ```

2. **Failover Procedures**
   - DNS failover to backup servers
   - Load balancer configuration updates
   - Database failover procedures

## 📋 Deployment Checklist

### Pre-deployment
- [ ] Code tested and reviewed
- [ ] Dependencies updated and secure
- [ ] Configuration files prepared
- [ ] SSL certificates obtained
- [ ] Backup procedures tested
- [ ] Monitoring configured

### Deployment
- [ ] Application deployed
- [ ] Database migrations applied
- [ ] Services started and configured
- [ ] Load balancer configured
- [ ] DNS records updated
- [ ] SSL certificates installed

### Post-deployment
- [ ] Health checks passing
- [ ] Monitoring alerts configured
- [ ] Performance metrics baseline established
- [ ] Backup procedures verified
- [ ] Security scan completed
- [ ] Documentation updated

---

For specific deployment questions or issues, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file or create an issue in the repository.
