/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        slate: {
          850: '#151e32',
          900: '#0f172a',
          950: '#020617',
        },
        primary: {
          DEFAULT: '#0ea5e9', // Sky 500
          glow: 'rgba(14, 165, 233, 0.5)',
        },
        danger: {
          DEFAULT: '#ef4444', // Red 500
          glow: 'rgba(239, 68, 68, 0.5)',
        }
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'monospace'],
        sans: ['"Inter"', 'sans-serif'],
      },
      boxShadow: {
        'neon': '0 0 10px theme("colors.primary.glow"), 0 0 20px theme("colors.primary.glow")',
        'neon-red': '0 0 10px theme("colors.danger.glow"), 0 0 20px theme("colors.danger.glow")',
      }
    },
  },
  plugins: [],
}