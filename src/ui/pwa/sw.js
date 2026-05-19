/**
 * PostureMonitor PWA — Service Worker
 *
 * Caches the PWA shell for offline use. Handles push events
 * (reserved for future Web Push API integration).
 *
 * Cache strategy: Cache-First for shell files, Network-First
 * for dynamic content. On install, pre-cache all shell assets.
 *
 * Universidad Surcolombiana, 2026
 */

/* global self, caches, fetch, Response */

const CACHE_NAME = 'posturemonitor-v1';

const SHELL_FILES = [
  '/pwa/',
  '/pwa/index.html',
  '/pwa/app.js',
  '/pwa/style.css',
  '/pwa/manifest.json',
  '/pwa/icon.svg',
];

// CDN resources used by the app
const EXTERNAL_CACHES = [
  'https://cdn.jsdelivr.net/npm/jsqr@1.4.0/dist/jsQR.min.js',
];

const ALL_CACHES = [CACHE_NAME];

// ── Install: pre-cache shell files ─────────────────────────────
self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      const results = await Promise.allSettled(
        SHELL_FILES.map(async (url) => {
          try {
            const response = await fetch(url);
            if (response.ok) {
              await cache.put(url, response);
            }
          } catch (err) {
            // File may not be available at install time (dev server)
          }
        })
      );
      const failed = results.filter((r) => r.status === 'rejected').length;
      if (failed > 0) {
        // eslint-disable-next-line no-console
        console.warn(`[SW] ${failed} shell file(s) failed to cache`);
      }
    })()
  );
  // Activate immediately — don't wait for page reload
  self.skipWaiting();
});

// ── Activate: clean old caches ─────────────────────────────────
self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((key) => !ALL_CACHES.includes(key))
          .map((key) => caches.delete(key))
      );
    })()
  );
  // Take control of all clients immediately
  self.clients.claim();
});

// ── Fetch: cache-first for shell, network-first for CDN ────────
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Only handle GET requests
  if (request.method !== 'GET') return;

  // Cache-First for our shell files
  if (url.origin === self.location.origin && url.pathname.startsWith('/pwa/')) {
    event.respondWith(cacheFirst(request));
    return;
  }

  // Cache-First for known CDN resources
  if (EXTERNAL_CACHES.includes(url.href)) {
    event.respondWith(cacheFirst(request));
    return;
  }

  // Network-First for everything else
  event.respondWith(networkFirst(request));
});

/**
 * Cache-First strategy: serve from cache if available,
 * otherwise fetch from network and cache the response.
 */
async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response.ok && response.type === 'basic') {
      const cache = await caches.open(CACHE_NAME);
      // Clone because response can only be consumed once
      cache.put(request, response.clone());
    }
    return response;
  } catch (err) {
    // Offline and not in cache — return a fallback
    if (request.destination === 'document') {
      const fallback = await caches.match('/pwa/index.html');
      if (fallback) return fallback;
    }
    return new Response('Offline', { status: 503, statusText: 'Service Unavailable' });
  }
}

/**
 * Network-First strategy: try network, fallback to cache.
 */
async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch (err) {
    const cached = await caches.match(request);
    if (cached) return cached;
    return new Response('Offline', { status: 503, statusText: 'Service Unavailable' });
  }
}

// ── Push Event (reserved for future Web Push API) ──────────────
self.addEventListener('push', (event) => {
  if (!event.data) return;

  try {
    const payload = event.data.json();

    const title = payload.title || 'PostureMonitor';
    const options = {
      body: payload.body || 'Alerta postural recibida',
      icon: '/pwa/icon.svg',
      badge: '/pwa/icon.svg',
      vibrate: [200, 100, 200],
      data: payload.data || {},
      tag: payload.tag || 'posture-alert',
      renotify: true,
      requireInteraction: true,
    };

    event.waitUntil(self.registration.showNotification(title, options));
  } catch (err) {
    // Payload wasn't JSON — show as plain text
    event.waitUntil(
      self.registration.showNotification('PostureMonitor', {
        body: event.data.text(),
        icon: '/pwa/icon.svg',
      })
    );
  }
});

// ── Notification Click → focus or open app ─────────────────
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  const urlToOpen = '/pwa/';

  event.waitUntil(
    (async () => {
      const clients = await self.clients.matchAll({
        type: 'window',
        includeUncontrolled: true,
      });

      // Focus existing window if any
      for (const client of clients) {
        if (client.url.includes('/pwa/') && 'focus' in client) {
          await client.focus();
          return;
        }
      }

      // Open new window
      if (self.clients.openWindow) {
        await self.clients.openWindow(urlToOpen);
      }
    })()
  );
});
