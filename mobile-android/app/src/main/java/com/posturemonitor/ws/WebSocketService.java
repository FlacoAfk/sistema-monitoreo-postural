package com.posturemonitor.ws;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Build;
import android.os.IBinder;

import androidx.annotation.Nullable;
import androidx.core.app.NotificationCompat;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import com.posturemonitor.MainActivity;
import com.posturemonitor.R;

import org.json.JSONObject;

/**
 * Foreground service that wraps WsClient to keep the WebSocket alive
 * when the app is backgrounded or closed. Runs with type dataSync on Android 15+.
 *
 * Lifecycle:
 *   onCreate() → create notification channel, startForeground()
 *   onStartCommand() → extract sid/wsUrl, connect WebSocket
 *   onDestroy() → disconnect WebSocket, stopForeground()
 *
 * The service uses START_STICKY so Android restarts it if killed.
 * Pairing (sid + wsUrl) is persisted in SharedPreferences by MainActivity,
 * so the service can reconnect after a restart.
 */
public class WebSocketService extends Service {

    private static final String PREFS_NAME = "posture_monitor_prefs";
    private static final String PREF_SID = "sid";
    private static final String PREF_WS_URL = "ws_url";

    // Intent actions
    public static final String ACTION_CONNECT = "com.posturemonitor.CONNECT";
    public static final String ACTION_DISCONNECT = "com.posturemonitor.DISCONNECT";
    public static final String ACTION_STOP = "com.posturemonitor.STOP";

    // Intent extras
    public static final String EXTRA_SID = "sid";
    public static final String EXTRA_WS_URL = "wsUrl";

    // Broadcast actions (sent via LocalBroadcastManager)
    public static final String BROADCAST_CONNECTED = "com.posturemonitor.WS_CONNECTED";
    public static final String BROADCAST_DISCONNECTED = "com.posturemonitor.WS_DISCONNECTED";
    public static final String BROADCAST_ALERT = "com.posturemonitor.WS_ALERT";
    public static final String BROADCAST_RESOLUTION = "com.posturemonitor.WS_RESOLUTION";
    public static final String BROADCAST_PERSON_LEFT = "com.posturemonitor.WS_PERSON_LEFT";
    public static final String BROADCAST_ERROR = "com.posturemonitor.WS_ERROR";

    // Extras for alert broadcast
    public static final String EXTRA_ALERT_JSON = "alertJson";
    public static final String EXTRA_PERSON_ID = "personId";
    public static final String EXTRA_ERROR_MSG = "errorMsg";

    // Notification channel (separate from posture alert channel)
    static final String FG_CHANNEL_ID = "ws_foreground";
    private static final int FG_NOTIFICATION_ID = 1;

    private WsClient wsClient;
    private LocalBroadcastManager broadcastManager;

    @Override
    public void onCreate() {
        super.onCreate();
        broadcastManager = LocalBroadcastManager.getInstance(this);
        try {
            createNotificationChannel();
        } catch (Exception ignored) {
        }
        try {
            startForeground(FG_NOTIFICATION_ID, buildNotification());
        } catch (Exception ignored) {
        }
        wsClient = new WsClient();
        wsClient.addListener(wsListener);
    }

    @Override
    public int onStartCommand(@Nullable Intent intent, int flags, int startId) {
        if (intent == null) {
            // Service restarted by START_STICKY — try to reconnect from saved prefs
            tryReconnectFromPrefs();
            return START_STICKY;
        }

        String action = intent.getAction();
        if (ACTION_CONNECT.equals(action)) {
            String sid = intent.getStringExtra(EXTRA_SID);
            String wsUrl = intent.getStringExtra(EXTRA_WS_URL);
            if (sid != null && wsUrl != null) {
                if (wsClient != null) {
                    wsClient.connect(sid, wsUrl);
                }
            }
        } else if (ACTION_DISCONNECT.equals(action)) {
            if (wsClient != null) {
                wsClient.disconnect();
            }
        } else if (ACTION_STOP.equals(action)) {
            if (wsClient != null) {
                wsClient.disconnect();
            }
            try {
                stopForeground(STOP_FOREGROUND_REMOVE);
            } catch (Exception ignored) {
            }
            stopSelf();
        }

        return START_STICKY;
    }

    /**
     * When the service is restarted by START_STICKY (intent == null),
     * read the saved pairing from SharedPreferences and reconnect.
     */
    private void tryReconnectFromPrefs() {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        String sid = prefs.getString(PREF_SID, null);
        String wsUrl = prefs.getString(PREF_WS_URL, null);
        if (sid != null && wsUrl != null && wsClient != null) {
            wsClient.connect(sid, wsUrl);
        }
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null; // Not a bound service
    }

    @Override
    public void onDestroy() {
        if (wsClient != null) {
            try {
                wsClient.disconnect();
            } catch (Exception ignored) {
            }
        }
        try {
            stopForeground(STOP_FOREGROUND_REMOVE);
        } catch (Exception ignored) {
        }
        super.onDestroy();
    }

    // ── WsClient.Listener ──────────────────────────────────────────────

    private final WsClient.Listener wsListener = new WsClient.Listener() {
        @Override
        public void onConnected() {
            try {
                broadcastManager.sendBroadcast(new Intent(BROADCAST_CONNECTED));
            } catch (Exception ignored) {
            }
        }

        @Override
        public void onDisconnected() {
            try {
                broadcastManager.sendBroadcast(new Intent(BROADCAST_DISCONNECTED));
            } catch (Exception ignored) {
            }
        }

        @Override
        public void onAlert(JSONObject alert) {
            try {
                Intent intent = new Intent(BROADCAST_ALERT);
                intent.putExtra(EXTRA_ALERT_JSON, alert.toString());
                broadcastManager.sendBroadcast(intent);
            } catch (Exception ignored) {
            }
        }

        @Override
        public void onResolution(int personId) {
            try {
                Intent intent = new Intent(BROADCAST_RESOLUTION);
                intent.putExtra(EXTRA_PERSON_ID, personId);
                broadcastManager.sendBroadcast(intent);
            } catch (Exception ignored) {
            }
        }

        @Override
        public void onPersonLeft(int personId) {
            try {
                Intent intent = new Intent(BROADCAST_PERSON_LEFT);
                intent.putExtra(EXTRA_PERSON_ID, personId);
                broadcastManager.sendBroadcast(intent);
            } catch (Exception ignored) {
            }
        }

        @Override
        public void onError(String message) {
            try {
                Intent intent = new Intent(BROADCAST_ERROR);
                intent.putExtra(EXTRA_ERROR_MSG, message);
                broadcastManager.sendBroadcast(intent);
            } catch (Exception ignored) {
            }
        }
    };

    // ── Foreground notification ────────────────────────────────────────

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    FG_CHANNEL_ID,
                    getString(R.string.ws_channel_name),
                    NotificationManager.IMPORTANCE_LOW
            );
            channel.setDescription("Canal para el servicio en primer plano de WebSocket");
            channel.setShowBadge(false);
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) {
                manager.createNotificationChannel(channel);
            }
        }
    }

    private Notification buildNotification() {
        Intent tapIntent = new Intent(this, MainActivity.class);
        tapIntent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        PendingIntent pendingIntent = PendingIntent.getActivity(
                this, 0, tapIntent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
        );

        return new NotificationCompat.Builder(this, FG_CHANNEL_ID)
                .setContentTitle(getString(R.string.app_name))
                .setContentText(getString(R.string.ws_notification_text))
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setOngoing(true)
                .setSilent(true)
                .setContentIntent(pendingIntent)
                .build();
    }
}
