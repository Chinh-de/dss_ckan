/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        background: '#090A0F',
        foreground: '#F1F5F9',
        card: {
          DEFAULT: '#11141E',
          foreground: '#F1F5F9',
        },
        popover: {
          DEFAULT: '#141824',
          foreground: '#F1F5F9',
        },
        primary: {
          DEFAULT: '#E5A93C',
          foreground: '#090A0F',
        },
        secondary: {
          DEFAULT: '#1E2333',
          foreground: '#E2E8F0',
        },
        muted: {
          DEFAULT: '#181C29',
          foreground: '#94A3B8',
        },
        accent: {
          DEFAULT: '#D9383A',
          foreground: '#FFFFFF',
        },
        destructive: {
          DEFAULT: '#EF4444',
          foreground: '#FFFFFF',
        },
        border: 'rgba(255, 255, 255, 0.08)',
        input: 'rgba(255, 255, 255, 0.1)',
        ring: '#E5A93C',
      },
      borderRadius: {
        lg: '12px',
        md: '8px',
        sm: '6px',
      },
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      boxShadow: {
        'glow-gold': '0 0 24px -4px rgba(229, 169, 60, 0.25)',
        'glow-crimson': '0 0 24px -4px rgba(217, 56, 58, 0.25)',
      },
    },
  },
  plugins: [],
};

