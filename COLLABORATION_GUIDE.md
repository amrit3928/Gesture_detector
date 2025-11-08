# Collaboration Guide

## How to Share the Repository and Allow Others to Edit

### Option 1: Add Collaborators (Recommended for Team Projects)

#### Step 1: Add Team Members as Collaborators

1. Go to your GitHub repository: https://github.com/amrit3928/Gesture_detector
2. Click on **Settings** (top right of the repository page)
3. In the left sidebar, click on **Collaborators**
4. Click **Add people** button
5. Enter the GitHub usernames of your team members:
   - Hank(Bohan) Fang's GitHub username
   - Chaoxiang Zhang's GitHub username
   - Sophie Koehler's GitHub username
6. Select permission level: **Write** (allows them to push changes)
7. Click **Add [username] to this repository**

#### Step 2: Team Members Clone the Repository

Each team member should run:

```bash
git clone https://github.com/amrit3928/Gesture_detector.git
cd Gesture_detector
```

#### Step 3: Team Members Set Up Their Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Option 2: Make Repository Public (Not Recommended for Class Projects)

If you make the repository public, anyone can view it, but you still need to add collaborators for editing.

1. Go to **Settings** → **General**
2. Scroll down to **Danger Zone**
3. Click **Change visibility** → **Make public**

**Note:** This is usually not recommended for class projects as it makes your code publicly visible.

---

## Workflow for Team Collaboration

### Basic Workflow

1. **Pull latest changes** before starting work:
   ```bash
   git pull origin main
   ```

2. **Create a branch** for your feature (optional but recommended):
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** to the code

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. **Push your changes**:
   ```bash
   git push origin main
   # Or if using a branch:
   git push origin feature/your-feature-name
   ```

### Recommended: Use Branches for Features

For better collaboration, each team member can work on a separate branch:

```bash
# Create a new branch
git checkout -b feature/hand-detector-implementation

# Make changes and commit
git add .
git commit -m "Implement hand detector with MediaPipe"

# Push branch to GitHub
git push origin feature/hand-detector-implementation
```

Then create a Pull Request on GitHub to merge into main.

---

## Team Member Roles (Suggested)

Based on your project structure, here are suggested roles:

### Amritpal Singh
- Project lead / Integration
- Main repository owner

### Hank(Bohan) Fang
- Could work on: `hand_detector.py` implementation
- MediaPipe integration

### Chaoxiang Zhang
- Could work on: `gesture_classifier.py` implementation
- Neural network model

### Sophie Koehler
- Could work on: `video_processor.py` implementation
- Video processing and visualization

---

## Best Practices

1. **Always pull before pushing**:
   ```bash
   git pull origin main
   git push origin main
   ```

2. **Write clear commit messages**:
   ```bash
   git commit -m "Add MediaPipe hand detection implementation"
   ```

3. **Communicate with team** about what you're working on

4. **Test your changes** before pushing

5. **Use branches** for major features to avoid conflicts

---

## Troubleshooting

### If you get "permission denied" errors:
- Make sure you've been added as a collaborator
- Check that you're using the correct GitHub username/password
- Consider using a Personal Access Token instead of password

### If you get merge conflicts:
```bash
git pull origin main
# Resolve conflicts in the files
git add .
git commit -m "Resolve merge conflicts"
git push origin main
```

### If you need to update your local repository:
```bash
git pull origin main
```

---

## Quick Reference

**Clone repository:**
```bash
git clone https://github.com/amrit3928/Gesture_detector.git
```

**Get latest changes:**
```bash
git pull origin main
```

**Push your changes:**
```bash
git add .
git commit -m "Your commit message"
git push origin main
```

**Check status:**
```bash
git status
```

