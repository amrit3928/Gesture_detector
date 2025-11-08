# Git Setup Instructions

## Connect Local Project to GitHub Repository "Gesture_detector"

### Step 1: Initialize Git (if not already done)
```bash
git init
```

### Step 2: Add all files
```bash
git add .
```

### Step 3: Make initial commit
```bash
git commit -m "Initial commit: Project skeleton for hand gesture recognition"
```

### Step 4: Add remote repository
Replace `YOUR_USERNAME` with your GitHub username:
```bash
git remote add origin https://github.com/YOUR_USERNAME/Gesture_detector.git
```

### Step 5: Rename branch to main (if needed)
```bash
git branch -M main
```

### Step 6: Push to GitHub
```bash
git push -u origin main
```

## Quick Copy-Paste (replace YOUR_USERNAME):
```bash
git init
git add .
git commit -m "Initial commit: Project skeleton for hand gesture recognition"
git remote add origin https://github.com/YOUR_USERNAME/Gesture_detector.git
git branch -M main
git push -u origin main
```

## If you get authentication errors:
- Use GitHub CLI: `gh auth login`
- Or use a Personal Access Token instead of password
- Or use SSH: `git remote add origin git@github.com:YOUR_USERNAME/Gesture_detector.git`

