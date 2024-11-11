import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 51440,
    proxy: {
      '/api': {
        target: 'http://localhost:51441',
        changeOrigin: true,
      }
    }
  }
});
