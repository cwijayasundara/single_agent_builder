# Deploying Configurable Agents API to Google Cloud Run

This guide provides comprehensive instructions for deploying the Configurable Agents API to Google Cloud Run, including setup, configuration, and best practices for production deployment.

## Prerequisites

- Google Cloud Platform account with billing enabled
- Google Cloud CLI (`gcloud`) installed and authenticated
- Docker installed locally
- API source code with proper configuration

## Quick Start

```bash
# 1. Clone and configure the project
git clone <your-repo-url>
cd single_agent_builder

# 2. Set up environment variables
cp env.example .env
# Edit .env with your API keys

# 3. Build and deploy
gcloud run deploy configurable-agents-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --env-vars-file .env.yaml
```

## Detailed Deployment Steps

### 1. Prepare Your Environment

#### Set up Google Cloud Project
```bash
# Set your project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

#### Create Environment Configuration
Create `.env.yaml` for Cloud Run environment variables:

```yaml
# .env.yaml
OPENAI_API_KEY: "your-openai-api-key"
ANTHROPIC_API_KEY: "your-anthropic-api-key"
GOOGLE_API_KEY: "your-google-api-key"
GROQ_API_KEY: "your-groq-api-key"

# Cloud Run specific
PORT: "8080"
ALLOWED_HOSTS: "*.run.app,your-custom-domain.com"

# Optional: Database and storage
DATABASE_URL: "postgresql://user:pass@host:port/db"
REDIS_URL: "redis://redis-host:6379"

# Security
SECRET_KEY: "your-secret-key-for-jwt"
API_KEY_SALT: "your-api-key-salt"

# Logging
LOG_LEVEL: "INFO"
LOG_FORMAT: "structured"
```

### 2. Create Dockerfile

Create a `Dockerfile` optimized for Cloud Run:

```dockerfile
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port (Cloud Run uses PORT environment variable)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1

# Run the application
CMD ["python", "run_api.py", "--host", "0.0.0.0", "--port", "8080"]
```

### 3. Update API Configuration for Cloud Run

Update `src/api/main.py` for Cloud Run compatibility:

```python
# Add to create_app() function
def create_app() -> FastAPI:
    # ... existing code ...
    
    # Cloud Run health check endpoint
    @app.get("/api/health", tags=["health"])
    async def health_check():
        """Health check endpoint for Cloud Run."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "production")
        }
    
    # Configure for Cloud Run
    port = int(os.getenv("PORT", 8080))
    
    return app

# Update server runner for Cloud Run
def run_server():
    """Run server for Cloud Run."""
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True
    )
```

### 4. Deploy to Cloud Run

#### Option A: Deploy from Source (Recommended)
```bash
gcloud run deploy configurable-agents-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --env-vars-file .env.yaml \
  --memory 2Gi \
  --cpu 2 \
  --timeout 900 \
  --concurrency 100 \
  --min-instances 0 \
  --max-instances 10
```

#### Option B: Deploy from Container Image
```bash
# Build and push to Artifact Registry
gcloud artifacts repositories create configurable-agents \
  --repository-format docker \
  --location us-central1

# Build image
docker build -t gcr.io/$PROJECT_ID/configurable-agents-api .

# Push image  
docker push gcr.io/$PROJECT_ID/configurable-agents-api

# Deploy
gcloud run deploy configurable-agents-api \
  --image gcr.io/$PROJECT_ID/configurable-agents-api \
  --platform managed \
  --region us-central1 \
  --env-vars-file .env.yaml
```

### 5. Configure Custom Domain (Optional)

```bash
# Map custom domain
gcloud run domain-mappings create \
  --service configurable-agents-api \
  --domain api.yourdomain.com \
  --region us-central1
```

## Production Configuration

### Security Best Practices

1. **Authentication**: Enable Cloud IAM authentication for production:
```bash
gcloud run deploy configurable-agents-api \
  --no-allow-unauthenticated \
  --region us-central1
```

2. **Service Account**: Create dedicated service account:
```bash
gcloud iam service-accounts create configurable-agents-sa \
  --display-name "Configurable Agents Service Account"

gcloud run services update configurable-agents-api \
  --service-account configurable-agents-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --region us-central1
```

3. **VPC Connector**: For database access (if using Cloud SQL):
```bash
gcloud run services update configurable-agents-api \
  --vpc-connector your-vpc-connector \
  --region us-central1
```

### Monitoring and Logging

1. **Enable structured logging** in `.env.yaml`:
```yaml
LOG_FORMAT: "structured"
LOG_LEVEL: "INFO"
```

2. **Set up monitoring** with Cloud Monitoring:
```bash
# Service will automatically send logs to Cloud Logging
# Metrics available in Cloud Monitoring console
```

### Environment-Specific Configuration

Create different configurations for staging and production:

```yaml
# staging.env.yaml
ENVIRONMENT: "staging"
LOG_LEVEL: "DEBUG"
ALLOWED_HOSTS: "staging-api.yourdomain.com,*.run.app"

# production.env.yaml  
ENVIRONMENT: "production"
LOG_LEVEL: "INFO"
ALLOWED_HOSTS: "api.yourdomain.com"
```

## Performance Optimization

### Resource Configuration
```bash
# High-performance configuration
gcloud run deploy configurable-agents-api \
  --memory 4Gi \
  --cpu 4 \
  --timeout 3600 \
  --concurrency 50 \
  --min-instances 1 \
  --max-instances 20
```

### Auto-scaling Settings
- **Min instances**: Set to 1+ for production to avoid cold starts
- **Max instances**: Based on expected load and quotas
- **Concurrency**: 50-100 for CPU-intensive AI operations

## Troubleshooting

### Common Issues

1. **Cold Starts**: Set min-instances > 0 or implement warming strategies
2. **Memory Limits**: Increase memory allocation for large models
3. **Timeout**: Increase timeout for long-running agent operations
4. **API Keys**: Ensure all required environment variables are set

### Debugging Commands
```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision" --limit 50

# Check service status
gcloud run services describe configurable-agents-api --region us-central1

# Test deployment
curl https://your-service-url/api/health
```

## Cost Optimization

1. **CPU Allocation**: Use CPU throttling for cost savings
2. **Min Instances**: Set to 0 for development, 1+ for production
3. **Request Timeout**: Optimize based on typical agent execution time
4. **Regional Deployment**: Choose region closest to users

## Continuous Deployment

Set up automated deployment with Cloud Build:

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'configurable-agents-api'
      - '--source=.'
      - '--platform=managed'
      - '--region=us-central1'
      - '--env-vars-file=.env.yaml'
```

## Additional Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [FastAPI Cloud Run Guide](https://cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-service)
- [Cloud Run Best Practices](https://cloud.google.com/run/docs/best-practices)
- [Container Runtime Contract](https://cloud.google.com/run/docs/container-contract)

## Support

For deployment issues:
1. Check Cloud Run logs in Google Cloud Console
2. Verify environment variables and API keys
3. Test endpoints with provided curl commands
4. Review FastAPI documentation at `/docs` endpoint