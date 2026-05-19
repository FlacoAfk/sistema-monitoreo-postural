package com.posturemonitor.ws;

import android.os.Handler;
import android.os.Looper;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;

/**
 * Cliente WebSocket para conectar al servidor de alertas posturales.
 * Protocolo:
 *   PWA → Server: {"type": "pair", "sid": "<uuid>"}
 *   Server → PWA: {"type": "paired", "sid": "<uuid>"}
 *   Server → PWA: {"type": "alert", "person_id": N, ...}
 *   Server → PWA: {"type": "resolution", "person_id": N}
 *   Server → PWA: {"type": "person_left", "person_id": N}
 *   Server → PWA: {"type": "ping"}
 *   PWA → Server: {"type": "pong"}
 */
public class WsClient {

    private static final long RECONNECT_DELAY_MS = 3000;
    private static final long MAX_RECONNECT_DELAY_MS = 30000;

    private final OkHttpClient httpClient;
    private WebSocket webSocket;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private String sid;
    private String wsUrl;
    private boolean connected;
    private long reconnectDelay = RECONNECT_DELAY_MS;

    private final List<Listener> listeners = new ArrayList<>();

    public interface Listener {
        void onConnected();
        void onDisconnected();
        void onAlert(JSONObject alert);
        void onResolution(int personId);
        void onPersonLeft(int personId);
        void onError(String message);
    }

    public WsClient() {
        httpClient = new OkHttpClient.Builder()
                .pingInterval(30, TimeUnit.SECONDS)
                .build();
    }

    public void addListener(Listener listener) {
        if (!listeners.contains(listener)) {
            listeners.add(listener);
        }
    }

    public void removeListener(Listener listener) {
        listeners.remove(listener);
    }

    public boolean isConnected() {
        return connected;
    }

    public void connect(String sid, String wsUrl) {
        // Idempotent: skip if already connected to the same endpoint
        if (connected && this.sid != null && this.wsUrl != null
                && this.sid.equals(sid) && this.wsUrl.equals(wsUrl)) {
            return;
        }
        // If connected to a different endpoint, disconnect cleanly first
        if (connected || webSocket != null) {
            internalDisconnect();
        }
        this.sid = sid;
        this.wsUrl = wsUrl;
        this.reconnectDelay = RECONNECT_DELAY_MS;
        connectInternal();
    }

    private void connectInternal() {
        // Clear any stale reconnect callbacks before creating a new connection
        mainHandler.removeCallbacksAndMessages(null);

        if (webSocket != null) {
            webSocket.cancel();
            webSocket = null;
        }

        Request request = new Request.Builder().url(wsUrl).build();
        webSocket = httpClient.newWebSocket(request, new WebSocketListener() {
            @Override
            public void onOpen(WebSocket ws, Response response) {
                sendPair();
            }

            @Override
            public void onMessage(WebSocket ws, String text) {
                handleMessage(text);
            }

            @Override
            public void onClosed(WebSocket ws, int code, String reason) {
                setConnected(false);
                scheduleReconnect();
            }

            @Override
            public void onFailure(WebSocket ws, Throwable t, Response response) {
                setConnected(false);
                scheduleReconnect();
            }
        });
    }

    private void sendPair() {
        try {
            JSONObject msg = new JSONObject();
            msg.put("type", "pair");
            msg.put("sid", sid);
            webSocket.send(msg.toString());
        } catch (JSONException ignored) {
        }
    }

    private void handleMessage(String text) {
        try {
            JSONObject json = new JSONObject(text);
            String type = json.optString("type", "");

            switch (type) {
                case "paired":
                    setConnected(true);
                    reconnectDelay = RECONNECT_DELAY_MS;
                    break;
                case "alert":
                    notifyAlert(json);
                    break;
                case "resolution":
                    int personId = json.optInt("person_id", -1);
                    notifyResolution(personId);
                    break;
                case "person_left":
                    int leftId = json.optInt("person_id", -1);
                    notifyPersonLeft(leftId);
                    break;
                case "ping":
                    sendPong();
                    break;
            }
        } catch (JSONException ignored) {
        }
    }

    private void sendPong() {
        if (webSocket != null && connected) {
            try {
                JSONObject msg = new JSONObject();
                msg.put("type", "pong");
                webSocket.send(msg.toString());
            } catch (JSONException ignored) {
            }
        }
    }

    private void scheduleReconnect() {
        mainHandler.postDelayed(() -> {
            if (!connected && sid != null && wsUrl != null) {
                connectInternal();
                reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY_MS);
            }
        }, reconnectDelay);
    }

    private void setConnected(boolean connected) {
        this.connected = connected;
        if (connected) {
            notifyConnected();
        } else {
            notifyDisconnected();
        }
    }

    private void notifyConnected() {
        mainHandler.post(() -> {
            for (Listener l : listeners) {
                l.onConnected();
            }
        });
    }

    private void notifyDisconnected() {
        mainHandler.post(() -> {
            for (Listener l : listeners) {
                l.onDisconnected();
            }
        });
    }

    private void notifyAlert(JSONObject alert) {
        mainHandler.post(() -> {
            for (Listener l : listeners) {
                l.onAlert(alert);
            }
        });
    }

    private void notifyResolution(int personId) {
        mainHandler.post(() -> {
            for (Listener l : listeners) {
                l.onResolution(personId);
            }
        });
    }

    private void notifyPersonLeft(int personId) {
        mainHandler.post(() -> {
            for (Listener l : listeners) {
                l.onPersonLeft(personId);
            }
        });
    }

    /**
     * Internal disconnect without resetting sid/wsUrl (used by reconnect logic).
     */
    private void internalDisconnect() {
        if (webSocket != null) {
            webSocket.cancel();
            webSocket = null;
        }
        connected = false;
        mainHandler.removeCallbacksAndMessages(null);
    }

    /**
     * Public disconnect — null-safe, resets all connection state.
     */
    public void disconnect() {
        internalDisconnect();
        this.sid = null;
        this.wsUrl = null;
    }
}
