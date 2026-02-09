import type { Express } from "express";
import { createServer, type Server } from "http";
import multer from "multer";
import { randomUUID } from "crypto";
import path from "path";
import fs from "fs";

const uploadDir = path.join(process.cwd(), "uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    cb(null, uploadDir);
  },
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `${randomUUID()}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: 300 * 1024 * 1024,
  },
  fileFilter: (_req, file, cb) => {
    const allowedMimes = [
      "audio/wav",
      "audio/x-wav",
      "audio/mpeg",
      "audio/mp3",
      "audio/mp4",
      "audio/x-m4a",
      "audio/m4a",
    ];
    const allowedExts = [".wav", ".mp3", ".m4a"];
    const ext = path.extname(file.originalname).toLowerCase();

    if (allowedMimes.includes(file.mimetype) || allowedExts.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error("Unsupported file format. Please upload WAV, MP3, or M4A files."));
    }
  },
});

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  app.get("/api/health", (_req, res) => {
    res.json({ status: "ok", timestamp: new Date().toISOString() });
  });

  // Simple upload endpoint — stores the file and returns the URL.
  // All audio analysis now happens client-side via Web Audio API.
  app.post("/api/upload", upload.single("audio"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          success: false,
          error: "No audio file provided",
        });
      }

      const filename = req.file.filename;
      console.log(`Audio file uploaded: ${filename}`);

      res.json({
        success: true,
        data: {
          audioUrl: `/uploads/${filename}`,
        },
      });
    } catch (error) {
      console.error("Error uploading audio:", error);
      res.status(500).json({
        success: false,
        error: error instanceof Error ? error.message : "Failed to upload audio",
      });
    }
  });

  // Serve uploaded audio files
  app.use("/uploads", (req, res, next) => {
    const cleanPath = req.path.replace(/^\//, "");
    const filePath = path.resolve(uploadDir, cleanPath);
    // Prevent path traversal: resolved path must be inside uploadDir
    if (!filePath.startsWith(uploadDir + path.sep) && filePath !== uploadDir) {
      return res.status(403).json({ error: "Forbidden" });
    }
    if (fs.existsSync(filePath)) {
      const ext = path.extname(filePath).toLowerCase();
      const mimeTypes: Record<string, string> = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
      };
      res.setHeader("Content-Type", mimeTypes[ext] || "application/octet-stream");
      res.sendFile(filePath);
    } else {
      next();
    }
  });

  return httpServer;
}
