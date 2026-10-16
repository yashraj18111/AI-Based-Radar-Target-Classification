# 🚀 Deployment Guide: Render (Backend) & Vercel (Frontend)

This guide walks you through deploying your **AI-Based Radar Target Classification** full-stack MERN application live to the web using **Render** for the Express + PyTorch ML backend and **Vercel** for the React frontend.

---

## 🏗️ Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │             User's Browser             │
                      └───────────────────┬────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
         ┌─────────────────────┐                     ┌─────────────────────┐
         │   Vercel Frontend   │                     │    Google Drive     │
         │   (React + Vite)    │                     │  Radar CSV Datasets │
         └──────────┬──────────┘                     └──────────┬──────────┘
                    │                                           │
                    │ REST API calls (Predict / History / Drive)│
                    ▼                                           │
         ┌──────────────────────────────────────────────────────┴──┐
         │                     Render Backend                      │
         │     Node.js + Express + PyTorch CNN Microservice        │
         │                                                         │
         │  - Multer CSV Upload                                    │
         │  - Google Drive Fetcher (`/api/drive-predict`)          │
         │  - Spectrogram Synthesis (Viridis 128×128 RGB)          │
         │  - PyTorch 2D-CNN Inference                             │
         └──────────┬──────────────────────────────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │    MongoDB Atlas    │
         │ Classification Logs │
         └─────────────────────┘
```

---

## Part 1: Deploy Backend on Render

Render will host your Express API server and execute the PyTorch radar classifier using Docker.

### Step 1: Push Your Code to GitHub
Ensure your latest code with `server/`, `client/`, `Dockerfile`, and `render.yaml` is committed and pushed:
```bash
git add .
git commit -m "Add MERN stack and Render/Vercel deployment configs"
git push origin main
```

### Step 2: Create a Free MongoDB Atlas Database (Optional but Recommended)
1. Go to [mongodb.com/atlas](https://www.mongodb.com/atlas) and sign in.
2. Create a free **M0 Shared Cluster**.
3. Under **Database Access**, create a user (e.g., `radar_admin`) and password.
4. Under **Network Access**, click **Add IP Address** -> Select **Allow Access from Anywhere** (`0.0.0.0/0`).
5. Click **Connect** -> **Drivers** -> Copy your connection string:
   ```
   mongodb+srv://<username>:<password>@cluster0.mongodb.net/radar_db?retryWrites=true&w=majority
   ```
   *(Note: If you skip MongoDB, the server automatically runs in in-memory mode so the app never crashes!)*

### Step 3: Deploy Web Service on Render
1. Go to [dashboard.render.com](https://dashboard.render.com) and sign in.
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository: `AI-Based-Radar-Target-Classification`.
4. Configure the settings:
   - **Name:** `radar-target-api` (or your choice)
   - **Region:** Any (e.g., Oregon or Frankfurt)
   - **Runtime:** **Docker**
   - **Plan:** **Free**
5. Add Environment Variables under **Advanced**:
   | Key | Value | Notes |
   |-----|-------|-------|
   | `PORT` | `5000` | Port for Express |
   | `NODE_ENV` | `production` | Production mode |
   | `PYTHON_CMD` | `python3` | Linux Python binary |
   | `MONGODB_URI` | `mongodb+srv://...` | Your Atlas connection string |
   | `GOOGLE_DRIVE_FOLDER_URL` | `https://drive.google.com/...` | Your Google Drive dataset folder link |
6. Click **Deploy Web Service**.
7. Once build completes, copy your live backend URL (e.g. `https://radar-target-api.onrender.com`).

---

## Part 2: Deploy Frontend on Vercel ("vel")

Vercel will host your high-speed React dashboard.

### Step 1: Connect Repository to Vercel
1. Go to [vercel.com](https://vercel.com) and log in with GitHub.
2. Click **Add New...** -> **Project**.
3. Import your `AI-Based-Radar-Target-Classification` repository.

### Step 2: Configure Project Settings
In the Vercel project configuration screen:
- **Framework Preset:** `Vite`
- **Root Directory:** Click `Edit` and select **`client`**
- **Build Command:** `npm run build`
- **Output Directory:** `dist`

### Step 3: Set Environment Variable
Expand the **Environment Variables** section and add:
| Name | Value |
|------|-------|
| `VITE_API_BASE_URL` | `https://radar-target-api.onrender.com` *(Replace with your actual Render URL from Part 1)* |

### Step 4: Click Deploy
1. Click **Deploy**.
2. Within 30 seconds, your site will be live at `https://your-project.vercel.app`! 🎉

---

## Part 3: Setting Up Google Drive Test Data

The main page includes a dedicated **Google Drive Test Data Repository Box** where users can:
1. Open and download sample CSVs directly from Google Drive.
2. Paste any shared Google Drive file link directly into the input box to test inference.

### How to configure your Google Drive link:
1. Create a folder in Google Drive (e.g., `Radar Signal Test Samples`).
2. Upload sample CSVs (you can use samples from `dataset/raw/Cars/`, `dataset/raw/Drones/`, and `dataset/raw/People/`).
3. Right-click the folder -> **Share** -> Change general access to **"Anyone with the link can view"**.
4. Copy the link (e.g., `https://drive.google.com/drive/folders/1ABCxyz...`).
5. Set `GOOGLE_DRIVE_FOLDER_URL` in your Render Environment Variables to update the link shown on the website!

### Testing via Google Drive in the Web UI:
- **Option A (One-Click):** Click any of the Quick Download buttons (`Car_Sample.csv`, `Drone_Sample.csv`, `Person_Sample.csv`) inside the Google Drive box and drag the downloaded file into the dropzone.
- **Option B (Direct Drive Link):** Share any radar CSV on Google Drive (permission: "Anyone with link can view"), paste the link into the **"Input Google Drive File Link"** box on the web page, and click **Fetch from Drive & Classify**!

---

## Part 4: Local Testing Before Deploying

You can run both backend and frontend locally:

### 1. Start Express Backend
```bash
cd server
npm install
node server.js
```
Backend runs at `http://localhost:5000`.

### 2. Start React Frontend
In a new terminal:
```bash
cd client
npm install
npm run dev
```
Frontend runs at `http://localhost:3000` with hot-reloading and automatic proxying to `http://localhost:5000`.

---

## 🎯 Verification Checklist

- [x] PyTorch model auto-downloads if missing
- [x] React frontend builds cleanly to `client/dist`
- [x] Express backend handles uploads, Google Drive downloads, and preset tests
- [x] MongoDB saves classification history (with in-memory fallback)
- [x] Dockerfile & `render.yaml` ready for Render
- [x] `vercel.json` configured for Vercel SPA routing and asset caching
