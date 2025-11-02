# Quick Start Guide - Doodle Classifier

Choose your frontend and get started in minutes!

## 🚀 Fastest Way to Run

### Option 1: Modern Stack (Recommended - Vite + Shadcn)

**Terminal 1 - Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend-2
npm install
npm run dev
```

✅ Open **http://localhost:5173**

---

### Option 2: Create React App

**Terminal 1 - Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm start
```

✅ Open **http://localhost:3000**

---

## 📁 Project Structure

```
DoodlePredictor/
├── backend/                    # FastAPI backend
│   ├── main.py                # Prediction API
│   └── requirements.txt       # Python dependencies
├── frontend/                   # CRA frontend (simple)
│   └── src/
│       ├── App.js
│       └── App.css
├── frontend-2/                 # Vite frontend (modern)
│   └── src/
│       ├── components/ui/     # Shadcn components
│       ├── App.jsx
│       └── index.css
├── model.h5                   # Pre-trained model (14MB)
├── app.py                     # Original Streamlit app
└── SETUP.md                   # Detailed setup guide
```

## 🎯 What Each Option Offers

### frontend/ (Create React App)
- ✅ Simple setup
- ✅ Works perfectly
- ✅ Basic but functional UI

### frontend-2/ (Vite + Shadcn) ⭐ Recommended
- ⚡ Lightning fast dev server
- 🎨 Beautiful, modern UI
- ♿ Accessible components
- 📱 Fully responsive
- 🔥 Hot module replacement

## 🐛 Troubleshooting

### Backend Issues

**Model not found:**
```bash
# Make sure model.h5 is in the root directory
ls ../model.h5  # from backend/ directory
```

**Port already in use:**
```bash
# Use a different port
uvicorn main:app --reload --port 8001
# Update frontend fetch URL accordingly
```

### Frontend Issues

**Dependencies not installing:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**CORS errors:**
- Make sure backend is running first
- Check browser console for exact error
- Verify backend CORS settings in `backend/main.py`

### Prediction Issues

**Not drawing on canvas:**
- Click and drag on the canvas
- Make sure you're inside the white canvas area

**Poor predictions:**
- Draw clearer, simpler objects
- Try drawing common objects: circle, house, tree, cat
- Make sure drawing is centered

## 📊 Testing the App

### Try These Doodles:
1. **Circle** - Draw a simple circle
2. **House** - Draw a house with a triangle roof
3. **Cat** - Draw a simple cat face
4. **Tree** - Draw a tree with branches
5. **Star** - Draw a 5-pointed star

These are simple objects that the model predicts well!

## 🎨 Which Frontend Should I Use?

### Choose **frontend/** if:
- You want the simplest setup
- You're just testing the app
- You're familiar with CRA

### Choose **frontend-2/** if: ⭐
- You want a professional-looking app
- You care about performance
- You want to customize the UI
- You're building something production-ready

## 🔗 Useful Links

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Vite Docs](https://vitejs.dev/)
- [Shadcn UI](https://ui.shadcn.com/)
- [Tailwind CSS](https://tailwindcss.com/)
- [React Docs](https://react.dev/)

## 🎓 Next Steps

1. ✅ Get the app running
2. 🎨 Try drawing some doodles
3. 📖 Read `FRONTEND_COMPARISON.md` to understand the differences
4. 🚀 Customize the UI to your liking
5. 📦 Build for production when ready

## 💡 Pro Tips

1. **Backend**: The model loads at startup, so first prediction is fast
2. **Canvas**: Draw bigger and clearer for better predictions
3. **Debugging**: Check browser console and backend logs for errors
4. **Performance**: Use frontend-2 for the best experience

## ❓ Need Help?

1. Check the detailed `SETUP.md`
2. Read `FRONTEND_COMPARISON.md` for feature comparison
3. Look at `frontend-2/FRONTEND_README.md` for Vite-specific info
4. Check backend logs in the terminal

---

Made with ❤️ using FastAPI, React, and TensorFlow
