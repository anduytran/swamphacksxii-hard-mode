import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { AppProvider } from "./lib/app";
import { AuthProvider } from "./lib/auth";
import { applyDisplay, loadDisplay } from "./lib/display";
import "./styles.css";

applyDisplay(loadDisplay()); // index.html already did this before paint; this keeps them in step

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AuthProvider>
      <AppProvider>
        <App />
      </AppProvider>
    </AuthProvider>
  </StrictMode>,
);
