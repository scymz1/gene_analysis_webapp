// Central configuration for all URLs
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;
export const BASE_URL = process.env.NEXT_PUBLIC_BASE_URL;

// Add other URL-related configurations here
export const ENDPOINTS = {
  uploadCSV: `${API_BASE_URL}/api/upload-csv/`,
  analyze: `${API_BASE_URL}/api/analyze/`,
  // Add more endpoints as needed
};
