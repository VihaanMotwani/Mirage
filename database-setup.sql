-- Create the verification_logs table
CREATE TABLE verification_logs (
    id SERIAL PRIMARY KEY,
    source_type VARCHAR(10) NOT NULL, -- 'upload' or 'url'
    trust_score FLOAT NOT NULL,
    metadata_score FLOAT NOT NULL,
    reverse_image_score FLOAT NOT NULL,
    deepfake_score FLOAT NOT NULL,
    photoshop_score FLOAT NOT NULL,
    fact_check_score FLOAT NOT NULL,
    summary TEXT,
    timestamp TIMESTAMP NOT NULL,
    results_json JSONB NOT NULL, -- Store full results as JSON
    image_hash VARCHAR(64) -- Optional: Store image hash for duplicate detection
);

-- Create index for faster timestamp queries
CREATE INDEX idx_verification_logs_timestamp ON verification_logs(timestamp);

-- Create index for image hash lookup (if implemented)
CREATE INDEX idx_verification_logs_image_hash ON verification_logs(image_hash);

-- Create users table if you implement user accounts
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(100) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_login TIMESTAMP
);

-- Create user_verifications table to track user's verification history
CREATE TABLE user_verifications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    verification_id INTEGER REFERENCES verification_logs(id),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create index for user lookup
CREATE INDEX idx_user_verifications_user_id ON user_verifications(user_id);
