# 🚀 AWS Deployment Guide: Amazon Sales Regression Model

## 📋 Overview

This guide provides step-by-step instructions for deploying your Amazon sales regression model to AWS production environment. We'll cover multiple deployment options:

1. **AWS Lambda** (Serverless)
2. **Amazon ECS/Fargate** (Containerized)
3. **Amazon SageMaker** (Managed ML Platform)

---

## 🎯 Prerequisites

### **AWS Account Setup**
```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
# Enter your Access Key ID, Secret Access Key, Region, and Output format
```

### **Required AWS Services**
- **S3**: Model artifact storage
- **Lambda**: Serverless compute
- **API Gateway**: HTTP endpoints
- **CloudWatch**: Monitoring and logging
- **IAM**: Security and permissions

---

## 🏗️ Option 1: AWS Lambda Deployment (Recommended for Start)

### **Step 1: Prepare Model Artifacts**

```python
# In your Jupyter notebook
import joblib
import pickle

# Save your trained model
model_artifacts = {
    'model': your_trained_model,
    'scaler': your_scaler,
    'feature_names': feature_names,
    'model_info': {
        'created_date': datetime.now().isoformat(),
        'version': '1.0.0',
        'performance_metrics': your_metrics
    }
}

# Save using joblib (preferred for scikit-learn)
joblib.dump(model_artifacts, 'amazon_sales_model.joblib')

# Also save as pickle for compatibility
with open('amazon_sales_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)
```

### **Step 2: Create S3 Bucket**

```bash
# Create S3 bucket for model artifacts
aws s3 mb s3://amazon-sales-ml-models-$(date +%s)

# Upload model to S3
aws s3 cp amazon_sales_model.joblib s3://amazon-sales-ml-models-$(date +%s)/

# Verify upload
aws s3 ls s3://amazon-sales-ml-models-$(date +%s)/
```

### **Step 3: Create Lambda Function**

**Create `lambda_function.py`:**
```python
import json
import joblib
import numpy as np
from datetime import datetime
import boto3
import os

def lambda_handler(event, context):
    """AWS Lambda function for sales prediction"""
    
    try:
        # Load model from S3 (or package with Lambda)
        s3 = boto3.client('s3')
        bucket_name = os.environ.get('MODEL_BUCKET', 'amazon-sales-ml-models')
        model_key = os.environ.get('MODEL_KEY', 'amazon_sales_model.joblib')
        
        # Download model if not already cached
        model_path = '/tmp/model.joblib'
        s3.download_file(bucket_name, model_key, model_path)
        model_artifacts = joblib.load(model_path)
        
        # Parse input
        body = json.loads(event['body'])
        input_data = np.array(body['features']).reshape(1, -1)
        
        # Validate input
        if input_data.shape[1] != len(model_artifacts['feature_names']):
            raise ValueError(f"Expected {len(model_artifacts['feature_names'])} features, got {input_data.shape[1]}")
        
        # Make prediction
        prediction = model_artifacts['model'].predict(
            model_artifacts['scaler'].transform(input_data)
        )[0]
        
        # Prepare response
        response = {
            'prediction': float(prediction),
            'timestamp': datetime.now().isoformat(),
            'model_version': model_artifacts['model_info']['version'],
            'confidence': 0.95  # Add confidence score if available
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }
```

### **Step 4: Create Lambda Deployment Package**

```bash
# Create deployment directory
mkdir lambda_deployment
cd lambda_deployment

# Install dependencies
pip install -r ../requirements.txt -t .

# Copy Lambda function
cp ../lambda_function.py .

# Create deployment package
zip -r ../lambda_deployment.zip .
cd ..
```

### **Step 5: Deploy Lambda Function**

```bash
# Create IAM role for Lambda
aws iam create-role \
    --role-name lambda-sales-predictor-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

# Attach policies
aws iam attach-role-policy \
    --role-name lambda-sales-predictor-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
    --role-name lambda-sales-predictor-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Create Lambda function
aws lambda create-function \
    --function-name amazon-sales-predictor \
    --runtime python3.9 \
    --role arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/lambda-sales-predictor-role \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://lambda_deployment.zip \
    --timeout 30 \
    --memory-size 512 \
    --environment Variables='{MODEL_BUCKET=amazon-sales-ml-models,MODEL_KEY=amazon_sales_model.joblib}'
```

### **Step 6: Create API Gateway**

```bash
# Create REST API
aws apigateway create-rest-api \
    --name "Sales Prediction API" \
    --description "API for Amazon sales predictions"

# Get API ID
API_ID=$(aws apigateway get-rest-apis --query 'items[?name==`Sales Prediction API`].id' --output text)

# Get root resource ID
ROOT_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/`].id' --output text)

# Create resource
aws apigateway create-resource \
    --rest-api-id $API_ID \
    --parent-id $ROOT_ID \
    --path-part "predict"

# Get resource ID
RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[?path==`/predict`].id' --output text)

# Create POST method
aws apigateway put-method \
    --rest-api-id $API_ID \
    --resource-id $RESOURCE_ID \
    --http-method POST \
    --authorization-type NONE

# Set Lambda integration
aws apigateway put-integration \
    --rest-api-id $API_ID \
    --resource-id $RESOURCE_ID \
    --http-method POST \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri arn:aws:apigateway:$(aws configure get region):lambda:path/2015-03-31/functions/arn:aws:lambda:$(aws configure get region):$(aws sts get-caller-identity --query Account --output text):function:amazon-sales-predictor/invocations

# Deploy API
aws apigateway create-deployment \
    --rest-api-id $API_ID \
    --stage-name prod

# Get API URL
API_URL="https://$API_ID.execute-api.$(aws configure get region).amazonaws.com/prod/predict"
echo "API URL: $API_URL"
```

### **Step 7: Test the API**

```bash
# Test prediction endpoint
curl -X POST $API_URL \
    -H "Content-Type: application/json" \
    -d '{
        "features": [10000, 50000, 50, 0.1, 1, 25, 4.5, 45, 0.8, 0, -5, 0.2, 1000]
    }'
```

---

## 🐳 Option 2: Containerized Deployment (ECS/Fargate)

### **Step 1: Create Flask Application**

**Create `app.py`:**
```python
from flask import Flask, request, jsonify
import joblib
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model at startup
try:
    model_artifacts = joblib.load('model/amazon_sales_model.joblib')
    logger.info(f"Model loaded successfully: {model_artifacts['model_info']['version']}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model_artifacts = None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_artifacts is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Sales prediction endpoint"""
    if model_artifacts is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        
        # Validate input dimensions
        expected_features = len(model_artifacts['feature_names'])
        if features.shape[1] != expected_features:
            return jsonify({
                'error': f'Expected {expected_features} features, got {features.shape[1]}'
            }), 400
        
        # Make prediction
        prediction = model_artifacts['model'].predict(
            model_artifacts['scaler'].transform(features)
        )[0]
        
        # Log prediction
        logger.info(f"Prediction made: {prediction:.2f}")
        
        return jsonify({
            'prediction': float(prediction),
            'timestamp': datetime.now().isoformat(),
            'model_version': model_artifacts['model_info']['version'],
            'features_used': model_artifacts['feature_names']
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model_artifacts is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_info': model_artifacts['model_info'],
        'feature_names': model_artifacts['feature_names'],
        'model_type': type(model_artifacts['model']).__name__
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
```

### **Step 2: Create Dockerfile**

**Create `Dockerfile`:**
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY model/ ./model/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["python", "app.py"]
```

### **Step 3: Build and Push Docker Image**

```bash
# Build Docker image
docker build -t amazon-sales-predictor .

# Tag for ECR
aws ecr create-repository --repository-name amazon-sales-predictor

# Get ECR login token
aws ecr get-login-password --region $(aws configure get region) | \
    docker login --username AWS --password-stdin \
    $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$(aws configure get region).amazonaws.com

# Tag and push image
docker tag amazon-sales-predictor:latest \
    $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$(aws configure get region).amazonaws.com/amazon-sales-predictor:latest

docker push \
    $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$(aws configure get region).amazonaws.com/amazon-sales-predictor:latest
```

### **Step 4: Deploy to ECS Fargate**

**Create `task-definition.json`:**
```json
{
    "family": "amazon-sales-predictor",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "256",
    "memory": "512",
    "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "sales-predictor",
            "image": "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/amazon-sales-predictor:latest",
            "portMappings": [
                {
                    "containerPort": 8080,
                    "protocol": "tcp"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/amazon-sales-predictor",
                    "awslogs-region": "REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
                "interval": 30,
                "timeout": 5,
                "retries": 3,
                "startPeriod": 60
            }
        }
    ]
}
```

**Deploy to ECS:**
```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create ECS cluster
aws ecs create-cluster --cluster-name sales-prediction-cluster

# Create service
aws ecs create-service \
    --cluster sales-prediction-cluster \
    --service-name sales-predictor-service \
    --task-definition amazon-sales-predictor:1 \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

---

## 📊 Monitoring and Observability

### **CloudWatch Metrics**

```python
# Add CloudWatch metrics to your Lambda function
import boto3
import time

def log_metrics(prediction_time, prediction_value, error=None):
    """Log custom metrics to CloudWatch"""
    cloudwatch = boto3.client('cloudwatch')
    
    metrics = [
        {
            'MetricName': 'PredictionLatency',
            'Value': prediction_time,
            'Unit': 'Milliseconds'
        },
        {
            'MetricName': 'PredictionValue',
            'Value': prediction_value,
            'Unit': 'None'
        }
    ]
    
    if error:
        metrics.append({
            'MetricName': 'PredictionErrors',
            'Value': 1,
            'Unit': 'Count'
        })
    
    cloudwatch.put_metric_data(
        Namespace='AmazonSalesPredictor',
        MetricData=metrics
    )
```

### **Set Up Alarms**

```bash
# Create CloudWatch alarm for high error rate
aws cloudwatch put-metric-alarm \
    --alarm-name "SalesPredictor-HighErrorRate" \
    --alarm-description "High prediction error rate" \
    --metric-name Errors \
    --namespace AWS/Lambda \
    --statistic Sum \
    --period 300 \
    --threshold 5 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:REGION:ACCOUNT:TOPIC_NAME
```

---

## 🔄 CI/CD Pipeline

### **GitHub Actions Workflow**

**Create `.github/workflows/deploy.yml`:**
```yaml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Train and test model
      run: |
        python train_model.py
        python test_model.py
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Build and push Docker image
      run: |
        docker build -t amazon-sales-predictor .
        aws ecr get-login-password | docker login --username AWS --password-stdin ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.us-east-1.amazonaws.com
        docker tag amazon-sales-predictor:latest ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.us-east-1.amazonaws.com/amazon-sales-predictor:latest
        docker push ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.us-east-1.amazonaws.com/amazon-sales-predictor:latest
    
    - name: Deploy to ECS
      run: |
        aws ecs update-service --cluster sales-prediction-cluster --service sales-predictor-service --force-new-deployment
```

---

## 💰 Cost Optimization

### **Lambda Optimization**
- Use appropriate memory allocation (more memory = more CPU)
- Implement connection pooling for external services
- Use Lambda layers for common dependencies

### **ECS Optimization**
- Use Spot instances for non-critical workloads
- Implement auto-scaling based on CPU/memory usage
- Use Application Load Balancer for traffic distribution

### **Monitoring Costs**
```bash
# Check Lambda costs
aws ce get-cost-and-usage \
    --time-period Start=2023-01-01,End=2023-01-31 \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=SERVICE

# Check ECS costs
aws ce get-cost-and-usage \
    --time-period Start=2023-01-01,End=2023-01-31 \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=SERVICE \
    --filter '{"Dimensions": {"Key": "SERVICE","Values": ["Amazon Elastic Container Service"]}}'
```

---

## 🔒 Security Best Practices

### **IAM Policies**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::amazon-sales-ml-models/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData"
            ],
            "Resource": "*"
        }
    ]
}
```

### **VPC Configuration**
- Deploy Lambda functions in VPC for database access
- Use private subnets for ECS tasks
- Implement security groups with minimal required access

### **Data Encryption**
- Enable encryption at rest for S3 buckets
- Use HTTPS for API Gateway endpoints
- Encrypt model artifacts before uploading to S3

---

## 🚀 Production Checklist

- [ ] Model performance validated on test data
- [ ] Error handling implemented
- [ ] Logging and monitoring configured
- [ ] Security policies applied
- [ ] Cost monitoring enabled
- [ ] Backup and recovery procedures documented
- [ ] Team trained on deployment process
- [ ] Rollback procedures tested
- [ ] Performance benchmarks established
- [ ] Business stakeholders notified

---

**🎉 Congratulations! Your Amazon sales prediction model is now production-ready!**

**Next Steps:**
1. Monitor performance and business impact
2. Plan for model retraining schedule
3. Scale based on traffic patterns
4. Implement A/B testing for model improvements 