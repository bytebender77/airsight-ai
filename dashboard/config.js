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

// 🚀 CONFIGURATION
const PROD_URL = 'https://airsight-ai.onrender.com'; // REPLACE THIS after your Render deploy!
const LOCAL_URL = 'http://localhost:5050';

// Auto-detect environment: use PROD if on github.io or a public domain
const API_URL = (location.hostname === 'localhost' || location.hostname === '127.0.0.1')
    ? LOCAL_URL
    : PROD_URL;

window.AIRSIGHT_API = API_URL;
console.log("📡 AirSight AI connecting to:", window.AIRSIGHT_API);
