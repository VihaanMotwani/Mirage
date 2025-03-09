# Mirage - Image Verification API

Mirage is an advanced image verification system designed to analyze and assess the authenticity of images. It combines multiple detection techniques—including metadata analysis, reverse image search, deepfake detection, Photoshop manipulation detection, and fact checking—to compute an overall trust score for each submitted image.

## Table of Contents

- Features
- Directory Structure
- Installation
- Backend Setup
- Frontend Setup
- Usage
- API Endpoints
- Technologies Used
- Contributing
- License
- Notes

## Features

- **Multi-Layer Verification:**
  - Analyze images using a combination of:
    - **Metadata Analysis:** Extract and evaluate EXIF data for inconsistencies and anomalies.
    - **Reverse Image Search:** Locate the earliest published version of an image using RapidAPI.
    - **Deepfake Detection:** Leverage AI (via OpenAI API) to assess whether an image might be AI-generated.
    - **Photoshop Detection:** Detect signs of digital manipulation using error level analysis, noise consistency, DCT analysis, clone detection, and JPEG ghost detection.
    - **Fact Checking:** Validate claims and context using the Perplexity Sonar API.
- **Trust Score Calculation:**
  - Each component's result is weighted and combined into a single trust score, accompanied by a summary and key findings to help users interpret the outcome.
- **User-Friendly Frontend:**
  - A Next.js-based interface offers an intuitive image upload (or URL submission) and displays detailed verification results.
- **Robust API:**
  - Built with FastAPI, the backend provides endpoints for image verification, health checks, and verification history.

## Directory Structure

```
├── frontend/
│   ├── public/                   # Static assets (SVG icons)
│   ├── src/
│   │   ├── app/                  # Next.js pages and global styles
│   │   ├── components/           # React components (ImageUpload, ResultsDashboard)
│   │   └── lib/                  # API service functions
│   ├── .gitignore
│   ├── eslint.config.mjs
│   ├── next.config.ts
│   ├── package.json
│   ├── postcss.config.mjs
│   ├── README.md                 # Frontend README (this file)
│   └── tsconfig.json
├── services/                     # Python modules for image verification
│   ├── deepfake_detector.py
│   ├── fact_checker.py
│   ├── image_processor.py
│   ├── metadata_analyzer.py
│   ├── photoshop_detector.py
│   ├── reverse_image_search.py
│   └── trust_calculator.py
├── .gitignore
├── main.py                       # FastAPI server entry point
└── requirements.txt              # Python dependencies
```

## Installation

### Backend Setup

1. **Prerequisites:**

   - Python 3.8+
   - Virtual environment tool (optional but recommended)

2. **Clone the Repository:**

   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

3. **Create and Activate a Virtual Environment:**

   ```
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

4. **Install Dependencies:**

   ```
   pip install -r requirements.txt
   ```

5. **Configure Environment Variables:**
   Create a `.env` file in the project root with the following keys (replace placeholder values with your actual API keys and settings):
   ```
   OPENAI_API_KEY=your_openai_api_key
   PERPLEXITY_API_KEY=your_perplexity_api_key
   RAPID_API_KEY=your_rapidapi_key
   CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
   CLOUDINARY_API_KEY=your_cloudinary_api_key
   CLOUDINARY_API_SECRET=your_cloudinary_api_secret
   CORS_ORIGINS=http://localhost:3000
   ```
6. **Run the backend python file**

```bash
python main.py
```

### Frontend Setup

1. **Navigate to the Frontend Directory:**

   ```
   cd frontend
   ```

2. **Install Node Dependencies:**

   ```
   npm install
   ```

   Alternatively, you can use yarn or pnpm if preferred.

3. **Run the Development Server:**
   ```
   npm run dev
   ```
   Open http://localhost:3000 in your browser to view the application.

## Usage

### API

- **Health Check:**
  `GET /api/health`
  Returns the status and availability of verification services.
- **Image Verification:**
  `POST /api/verify`
  Accepts an image file (upload) or an image URL (via form data) and returns detailed verification results including:
  - Overall trust score
  - Component scores (metadata, reverse image, deepfake, Photoshop, fact check)
  - A summary and key findings
  - Detailed analysis results from each service
- **Verification History:**
  `GET /api/history`
  Retrieves recent verification records stored in memory.

### Frontend

- Use the web interface to upload an image or enter an image URL.
- View a dynamic dashboard that presents the verification analysis and trust score.
- The dashboard includes interactive elements to review detailed findings for each verification component.

## API Endpoints

- **Root Endpoint:**
  `GET /`
  Returns a welcome message for the API.
- **Health Check:**
  `GET /api/health`
- **Verify Image:**
  `POST /api/verify`
  - Parameters (form data):
    - `source_type`: "upload" or "url"
    - `image`: (file upload if source_type is "upload")
    - `image_url`: (string if source_type is "url")
- **Verification History:**
  `GET /api/history?limit=<number>`

## Technologies Used

- **Backend:** FastAPI, Python, aiohttp, Pillow, OpenCV, scikit-learn, exifread
- **Frontend:** Next.js, React, Tailwind CSS
- **APIs:** OpenAI API, Perplexity Sonar API, RapidAPI for reverse image search
- **Others:** Cloudinary for image hosting and processing

## Contributing

Contributions are welcome! If you have ideas for improvements or bug fixes, please open an issue or submit a pull request.

## License

This project is released under the MIT License. See the LICENSE file for details.

--
