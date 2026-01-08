// API configuration and utility functions

// Declare process.env for Next.js environment variables
declare const process: {
  env: {
    NEXT_PUBLIC_API_BASE?: string;
  };
};

// Get API base URL from environment variable
// This is automatically set by start_web.py based on config/main.yaml
// The .env.local file is auto-generated on startup with the correct backend port
export const API_BASE_URL = (() => {
  // Check if we have a valid environment variable
  const envBase = process.env.NEXT_PUBLIC_API_BASE;
  
  // Filter out placeholder values that weren't replaced
  const isValidEnvBase = envBase && 
    !envBase.includes("__NEXT_PUBLIC_API_BASE_PLACEHOLDER__") &&
    !envBase.includes("undefined");
  
  if (isValidEnvBase) {
    return envBase;
  }
  
  // In Docker/production, fall back to same-origin with port 8001
  if (typeof window !== "undefined") {
    // Browser environment - construct from current location
    // This ensures the frontend uses the same hostname the user is accessing
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const backendPort = 8001; // Default backend port
    const fallbackUrl = `${protocol}//${hostname}:${backendPort}`;
    console.warn(`NEXT_PUBLIC_API_BASE not set, using fallback: ${fallbackUrl}`);
    return fallbackUrl;
  }
  
  // Server-side rendering - use localhost:8001
  const fallbackUrl = "http://localhost:8001";
  console.warn(`NEXT_PUBLIC_API_BASE not set (SSR), using fallback: ${fallbackUrl}`);
  return fallbackUrl;
})();

/**
 * Construct a full API URL from a path
 * @param path - API path (e.g., '/api/v1/knowledge/list')
 * @returns Full URL (e.g., 'http://localhost:8000/api/v1/knowledge/list')
 */
export function apiUrl(path: string): string {
  // Remove leading slash if present to avoid double slashes
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  // Remove trailing slash from base URL if present
  const base = API_BASE_URL.endsWith("/")
    ? API_BASE_URL.slice(0, -1)
    : API_BASE_URL;

  return `${base}${normalizedPath}`;
}

/**
 * Construct a WebSocket URL from a path
 * @param path - WebSocket path (e.g., '/api/v1/solve')
 * @returns WebSocket URL (e.g., 'ws://localhost:{backend_port}/api/v1/solve')
 * Note: backend_port is configured in config/main.yaml
 */
export function wsUrl(path: string): string {
  // Security Hardening: Convert http to ws and https to wss.
  // In production environments (where API_BASE_URL starts with https), this ensures secure websockets.
  const base = API_BASE_URL.replace(/^http:/, "ws:").replace(/^https:/, "wss:");

  // Remove leading slash if present to avoid double slashes
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  // Remove trailing slash from base URL if present
  const normalizedBase = base.endsWith("/") ? base.slice(0, -1) : base;

  return `${normalizedBase}${normalizedPath}`;
}
