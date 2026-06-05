/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "monospace",
        ],
      },
      colors: {
        ink: {
          900: "#0B1220",
          800: "#111A2C",
          700: "#1A2540",
          600: "#243152",
          500: "#3A4868",
        },
        accent: {
          DEFAULT: "#2563EB",
          hover: "#1D4ED8",
        },
      },
      boxShadow: {
        panel: "0 1px 2px rgba(15,23,42,0.04), 0 1px 3px rgba(15,23,42,0.06)",
        ring: "0 0 0 4px rgba(37,99,235,0.15)",
      },
    },
  },
  plugins: [],
};
