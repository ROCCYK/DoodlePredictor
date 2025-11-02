# Doodle Classifier - Modern Frontend (Vite + React + Shadcn)

This is the enhanced frontend built with modern technologies: **Vite**, **React**, **Tailwind CSS**, and **Shadcn UI**.

## Tech Stack

- ⚡ **Vite** - Fast build tool and dev server
- ⚛️ **React 19** - Latest React version
- 🎨 **Tailwind CSS** - Utility-first CSS framework
- 🎭 **Shadcn UI** - Beautiful, accessible UI components
- 🎯 **Lucide React** - Modern icon library
- 🖌️ **React Signature Canvas** - Canvas drawing component

## Features

- Modern, responsive UI with gradient designs
- Shadcn UI components (Button, Card, Badge)
- Real-time canvas drawing with smooth interactions
- Loading states with animated spinner
- Error handling with styled alerts
- Beautiful gradient backgrounds and borders
- Fully responsive layout (mobile-friendly)
- Dark mode support (via Shadcn theming)

## Setup Instructions

### 1. Navigate to the frontend-2 directory

```bash
cd frontend-2
```

### 2. Install dependencies (if not already done)

```bash
npm install
```

### 3. Start the development server

```bash
npm run dev
```

The app will run at **http://localhost:5173** (Vite's default port)

## Project Structure

```
frontend-2/
├── src/
│   ├── components/
│   │   └── ui/            # Shadcn UI components
│   │       ├── button.jsx
│   │       ├── card.jsx
│   │       └── badge.jsx
│   ├── lib/
│   │   └── utils.js       # Utility functions (cn)
│   ├── App.jsx            # Main application
│   ├── index.css          # Tailwind + theme config
│   └── main.jsx           # Entry point
├── tailwind.config.js     # Tailwind configuration
├── postcss.config.js      # PostCSS configuration
├── vite.config.js         # Vite configuration
└── package.json
```

## Key Differences from frontend/

### Better Stack
- **Vite** instead of Create React App (faster builds, better DX)
- **Shadcn UI** instead of custom CSS (consistent, accessible components)
- **Tailwind CSS** instead of plain CSS (utility-first approach)
- **Lucide React** instead of emoji icons (professional icons)

### Enhanced Features
- Better component architecture
- More polished UI/UX
- Responsive design with Tailwind
- Accessible components (Shadcn)
- Better loading states
- Professional color scheme (purple/indigo gradient)

## How It Works

The prediction logic is **exactly the same** as the first frontend:

1. User draws on the canvas
2. Canvas is converted to PNG data URL
3. Image is sent to backend at `http://localhost:8000/predict`
4. Backend processes image with the exact Streamlit pipeline
5. Top 3 predictions are displayed

## Customization

### Change Color Scheme

Edit `src/index.css` to change the CSS variables:

```css
--primary: 262 83% 58%;  /* Purple */
--ring: 262 83% 58%;      /* Focus ring */
```

### Add More Components

Shadcn UI has many more components you can add:

```bash
npx shadcn-ui@latest add [component-name]
```

## Building for Production

```bash
npm run build
```

The optimized build will be in the `dist/` directory.

## Backend Connection

Make sure the FastAPI backend is running at `http://localhost:8000` before using the app.

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Performance

- Fast initial load with Vite
- Code splitting for optimal bundle size
- Canvas operations run smoothly
- Minimal re-renders with React hooks

## Troubleshooting

### Port Already in Use

If port 5173 is in use, Vite will automatically try the next available port.

### Canvas Not Drawing

Make sure you're clicking and dragging on the canvas. The canvas accepts both mouse and touch events.

### Predictions Not Working

1. Ensure backend is running at `http://localhost:8000`
2. Check browser console for CORS errors
3. Verify backend CORS settings include `http://localhost:5173`
