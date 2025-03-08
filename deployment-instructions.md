# Deployment Instructions

## Prerequisites

- Node.js v16+ and npm
- Python 3.9+
- PostgreSQL database
- API keys for:
  - SerpAPI for reverse image search
  - OpenAI API for DALL-E detection
  - Google Fact Check API

## Frontend Deployment (Vercel)

1. **Setup Environment Variables**

Create a `.env.local` file in your frontend directory:

```
NEXT_PUBLIC_API_URL=https://your-backend-api-url.com
```

2. **Install Dependencies**

```bash
cd frontend
npm install
```

3. **Build and Deploy**

```bash
# For testing locally
npm run dev

# For production build
npm run build

# Deploy to Vercel
vercel
```

4. **Configure Vercel Project**

- Connect your GitHub repository
- Set the environment variables in the Vercel dashboard
- Enable automatic deployments on push

## Backend Deployment (AWS/GCP/Fly.io)

1. **Setup Environment Variables**

Create a `.env` file in your backend directory:

```
DATABASE_URL=postgresql://username:password@localhost:5432/imageauth
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
SERP_API_KEY=your_serp_api_key
CORS_ORIGINS=https://your-frontend-domain.com
```

2. **Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Database Setup**

```bash
# Run the database migration script
psql -U postgres -d imageauth -f database/init.sql
```

5. **Deploy on AWS Elastic Beanstalk**

```bash
# Install EB CLI
pip install awsebcli

# Initialize EB project
eb init

# Create environment and deploy
eb create production-environment
```

2. **Create Tables**

Run the SQL script provided in the `database-setup.sql` file.

3. **Set Up Database User**

```sql
CREATE USER imageauth_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE imageauth TO imageauth_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO imageauth_user;
```

## Monitoring and Maintenance

1. **Set Up Logging**

Configure logging with CloudWatch (AWS) or Datadog.

2. **Set Up Monitoring**

- Create health check endpoints
- Set up alerts for high error rates or API failures

3. **Backup Strategy**

- Set up automated database backups
- Store verification logs with appropriate retention policy
