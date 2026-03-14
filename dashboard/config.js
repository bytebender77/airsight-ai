/**
 * AirSight AI — API Configuration
 * ─────────────────────────────────
 * Change API_URL here when you get your ngrok URL.
 *
 * FOR LOCAL DEV:
 *   const API_URL = 'http://localhost:5050';
 *
 * FOR DEMO (replace with your ngrok URL each session):
 *   const API_URL = 'https://xxxx-xx-xx-xx-xx.ngrok-free.app';
 */

const API_URL = 'http://localhost:5050';

// Auto-detect: if dashboard is accessed over ngrok/public URL,
// you can override here or set window.API_URL before this loads.
window.AIRSIGHT_API = API_URL;
