import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import fs from "fs";

async function startServer() {
  const app = express();
  const PORT = 3000;
  const DATA_FILE = path.join(process.cwd(), "attendance.json");

  app.use(express.json());

  // Initialize data file if it doesn't exist
  if (!fs.existsSync(DATA_FILE)) {
    fs.writeFileSync(DATA_FILE, JSON.stringify({ users: [], attendance: [] }));
  }

  // API Routes
  app.get("/api/data", (req, res) => {
    const data = JSON.parse(fs.readFileSync(DATA_FILE, "utf-8"));
    res.json(data);
  });

  app.post("/api/users", (req, res) => {
    const { name, photo } = req.body;
    const data = JSON.parse(fs.readFileSync(DATA_FILE, "utf-8"));
    const newUser = { id: Date.now().toString(), name, photo, createdAt: new Date().toISOString() };
    data.users.push(newUser);
    fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2));
    res.json(newUser);
  });

  app.post("/api/attendance", (req, res) => {
    const { userId } = req.body;
    const data = JSON.parse(fs.readFileSync(DATA_FILE, "utf-8"));
    
    // Check if already marked today
    const today = new Date().toISOString().split('T')[0];
    const alreadyMarked = data.attendance.find(a => a.userId === userId && a.date.startsWith(today));
    
    if (alreadyMarked) {
      return res.status(400).json({ error: "Attendance already marked for today" });
    }

    const newRecord = { 
      id: Date.now().toString(), 
      userId, 
      date: new Date().toISOString(),
      status: "present"
    };
    data.attendance.push(newRecord);
    fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2));
    res.json(newRecord);
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
