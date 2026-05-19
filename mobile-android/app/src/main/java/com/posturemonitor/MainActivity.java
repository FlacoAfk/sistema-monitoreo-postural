package com.posturemonitor;

import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.PowerManager;
import android.provider.Settings;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.posturemonitor.model.PostureAlert;
import com.posturemonitor.ui.NotificationHelper;
import com.posturemonitor.ui.PersonCardAdapter;
import com.posturemonitor.ui.QrScannerActivity;
import com.posturemonitor.ws.WebSocketService;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * MainActivity — Pantalla principal de la app.
 * Maneja permisos runtime, inicia el WebSocketService y recibe
 * broadcasts del servicio para actualizar la UI.
 *
 * La vinculación (sid + wsUrl) se persiste en SharedPreferences,
 * así la app se auto-reconecta al abrirse si ya fue vinculada.
 * El WebSocketService corre como foreground service y sobrevive
 * aunque la Activity se destruya (background / app cerrada).
 */
public class MainActivity extends AppCompatActivity {

    private static final int QR_SCAN_REQUEST = 1;
    private static final int PERMISSION_REQUEST_CODE = 100;
    private static final String PREFS_NAME = "posture_monitor_prefs";
    private static final String PREF_SID = "sid";
    private static final String PREF_WS_URL = "ws_url";

    private NotificationHelper notificationHelper;
    private PersonCardAdapter adapter;
    private LocalBroadcastManager broadcastManager;
    private SharedPreferences prefs;

    // UI
    private TextView connectionStatus;
    private TextView statusDot;
    private Button scanQrButton;
    private Button disconnectButton;
    private RecyclerView personsRecyclerView;
    private TextView emptyStateText;
    private ProgressBar progressBar;
    private TextView sessionIdText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        notificationHelper = new NotificationHelper(this);
        adapter = new PersonCardAdapter();
        broadcastManager = LocalBroadcastManager.getInstance(this);
        prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);

        // Init UI
        connectionStatus = findViewById(R.id.connection_status);
        statusDot = findViewById(R.id.status_dot);
        scanQrButton = findViewById(R.id.btn_scan_qr);
        disconnectButton = findViewById(R.id.btn_disconnect);
        personsRecyclerView = findViewById(R.id.recycler_persons);
        emptyStateText = findViewById(R.id.empty_state);
        progressBar = findViewById(R.id.progress_bar);
        sessionIdText = findViewById(R.id.session_id);

        personsRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        personsRecyclerView.setAdapter(adapter);

        scanQrButton.setOnClickListener(v -> startQrScanner());
        disconnectButton.setOnClickListener(v -> disconnect());
        disconnectButton.setVisibility(View.GONE);

        updateDisconnected();

        // Check and request runtime permissions before allowing connections
        checkAndRequestPermissions();
    }

    // ── SharedPreferences ──────────────────────────────────────────

    private void savePairing(String sid, String wsUrl) {
        prefs.edit()
            .putString(PREF_SID, sid)
            .putString(PREF_WS_URL, wsUrl)
            .apply();
    }

    private void clearPairing() {
        prefs.edit()
            .remove(PREF_SID)
            .remove(PREF_WS_URL)
            .apply();
    }

    private boolean hasSavedPairing() {
        return prefs.contains(PREF_SID) && prefs.contains(PREF_WS_URL);
    }

    private String getSavedSid() {
        return prefs.getString(PREF_SID, null);
    }

    private String getSavedWsUrl() {
        return prefs.getString(PREF_WS_URL, null);
    }

    // ── Auto-reconnect ─────────────────────────────────────────────

    /**
     * Si hay vinculación guardada, reconecta automáticamente al abrir la app.
     */
    private void tryAutoReconnect() {
        if (!hasSavedPairing()) return;

        String sid = getSavedSid();
        String wsUrl = getSavedWsUrl();
        if (sid == null || wsUrl == null) return;

        progressBar.setVisibility(View.VISIBLE);
        scanQrButton.setVisibility(View.GONE);
        sessionIdText.setText("Sesión: " + sid.substring(0, Math.min(8, sid.length())) + "...");
        sessionIdText.setVisibility(View.VISIBLE);
        disconnectButton.setVisibility(View.VISIBLE);

        Intent serviceIntent = new Intent(this, WebSocketService.class);
        serviceIntent.setAction(WebSocketService.ACTION_CONNECT);
        serviceIntent.putExtra(WebSocketService.EXTRA_SID, sid);
        serviceIntent.putExtra(WebSocketService.EXTRA_WS_URL, wsUrl);
        ContextCompat.startForegroundService(this, serviceIntent);
    }

    // ── Permission Flow ──────────────────────────────────────────────

    private void checkAndRequestPermissions() {
        // Step 1: POST_NOTIFICATIONS (API 33+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.POST_NOTIFICATIONS},
                        PERMISSION_REQUEST_CODE + 0);
                return;
            }
        }

        // Step 2: CAMERA
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    PERMISSION_REQUEST_CODE + 1);
            return;
        }

        // All required permissions granted → check battery optimization
        checkBatteryOptimization();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (grantResults.length == 0) {
            Toast.makeText(this, "Permisos necesarios para el funcionamiento", Toast.LENGTH_LONG).show();
            return;
        }

        boolean granted = grantResults[0] == PackageManager.PERMISSION_GRANTED;

        int baseCode = requestCode - PERMISSION_REQUEST_CODE;

        if (baseCode == 0) {
            // POST_NOTIFICATIONS result → proceed to CAMERA
            if (!granted) {
                Toast.makeText(this, "Permiso de notificaciones denegado", Toast.LENGTH_SHORT).show();
            }
            checkCameraPermission();
        } else if (baseCode == 1) {
            // CAMERA result
            if (!granted) {
                Toast.makeText(this, "Permiso de cámara denegado — no se puede escanear QR", Toast.LENGTH_SHORT).show();
            }
            checkBatteryOptimization();
        }
    }

    private void checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    PERMISSION_REQUEST_CODE + 1);
            return;
        }
        checkBatteryOptimization();
    }

    // ── Battery Optimization ──────────────────────────────────────────

    private void checkBatteryOptimization() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
            if (pm != null && !pm.isIgnoringBatteryOptimizations(getPackageName())) {
                showBatteryOptimizationDialog();
            }
        }

        // After permissions + battery check, try auto-reconnect if previously paired
        tryAutoReconnect();
    }

    private void showBatteryOptimizationDialog() {
        new AlertDialog.Builder(this)
                .setTitle(getString(R.string.battery_opt_title))
                .setMessage(getString(R.string.battery_opt_message))
                .setPositiveButton(getString(R.string.battery_opt_positive), (dialog, which) -> {
                    try {
                        Intent intent = new Intent(
                                Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                                Uri.parse("package:" + getPackageName())
                        );
                        startActivity(intent);
                    } catch (Exception e) {
                        Intent fallback = new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS);
                        startActivity(fallback);
                    }
                })
                .setNegativeButton(getString(R.string.battery_opt_negative), null)
                .show();
    }

    // ── QR Scanner ────────────────────────────────────────────────────

    private void startQrScanner() {
        Intent intent = new Intent(this, QrScannerActivity.class);
        startActivityForResult(intent, QR_SCAN_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == QR_SCAN_REQUEST && resultCode == RESULT_OK && data != null) {
            String qrData = data.getStringExtra("qr_data");
            if (qrData != null) {
                handleQrData(qrData);
            }
        }
    }

    private void handleQrData(String qrData) {
        try {
            JSONObject json = new JSONObject(qrData);
            String sid = json.optString("sid", "");
            String wsUrl = json.optString("ws", "");

            if (sid.isEmpty() || wsUrl.isEmpty()) {
                Toast.makeText(this, R.string.error_invalid_qr, Toast.LENGTH_SHORT).show();
                return;
            }

            // Persist pairing
            savePairing(sid, wsUrl);

            progressBar.setVisibility(View.VISIBLE);
            scanQrButton.setVisibility(View.GONE);
            sessionIdText.setText("Sesión: " + sid.substring(0, Math.min(8, sid.length())) + "...");
            sessionIdText.setVisibility(View.VISIBLE);
            disconnectButton.setVisibility(View.VISIBLE);

            // Start WebSocketService with connection params
            Intent serviceIntent = new Intent(this, WebSocketService.class);
            serviceIntent.setAction(WebSocketService.ACTION_CONNECT);
            serviceIntent.putExtra(WebSocketService.EXTRA_SID, sid);
            serviceIntent.putExtra(WebSocketService.EXTRA_WS_URL, wsUrl);
            ContextCompat.startForegroundService(this, serviceIntent);
        } catch (JSONException e) {
            Toast.makeText(this, R.string.error_invalid_qr, Toast.LENGTH_SHORT).show();
        }
    }

    // ── Disconnect ────────────────────────────────────────────────────

    private void disconnect() {
        // Clear persisted pairing so we don't auto-reconnect next time
        clearPairing();

        Intent serviceIntent = new Intent(this, WebSocketService.class);
        serviceIntent.setAction(WebSocketService.ACTION_DISCONNECT);
        startService(serviceIntent);

        adapter.clear();
        emptyStateText.setVisibility(View.VISIBLE);
        scanQrButton.setVisibility(View.VISIBLE);
        disconnectButton.setVisibility(View.GONE);
        sessionIdText.setVisibility(View.GONE);
        updateDisconnected();
    }

    // ── Broadcast Receiver ────────────────────────────────────────────

    private final BroadcastReceiver wsReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (action == null) return;

            switch (action) {
                case WebSocketService.BROADCAST_CONNECTED:
                    runOnUiThread(MainActivity.this::updateConnected);
                    break;

                case WebSocketService.BROADCAST_DISCONNECTED:
                    runOnUiThread(() -> {
                        // Only update UI — service will auto-reconnect on its own
                        updateDisconnected();
                    });
                    break;

                case WebSocketService.BROADCAST_ALERT:
                    String alertStr = intent.getStringExtra(WebSocketService.EXTRA_ALERT_JSON);
                    if (alertStr != null) {
                        try {
                            JSONObject alertJson = new JSONObject(alertStr);
                            handleAlert(alertJson);
                        } catch (JSONException ignored) {
                        }
                    }
                    break;

                case WebSocketService.BROADCAST_RESOLUTION:
                    int personId = intent.getIntExtra(WebSocketService.EXTRA_PERSON_ID, -1);
                    runOnUiThread(() -> {
                        adapter.removePerson(personId);
                        notificationHelper.clearNotification(personId);
                        if (!adapter.hasPersons()) {
                            emptyStateText.setVisibility(View.VISIBLE);
                        }
                    });
                    break;

                case WebSocketService.BROADCAST_PERSON_LEFT:
                    int leftPersonId = intent.getIntExtra(WebSocketService.EXTRA_PERSON_ID, -1);
                    runOnUiThread(() -> {
                        adapter.removePerson(leftPersonId);
                        notificationHelper.clearNotification(leftPersonId);
                        if (!adapter.hasPersons()) {
                            emptyStateText.setVisibility(View.VISIBLE);
                        }
                    });
                    break;

                case WebSocketService.BROADCAST_PERSONS_UPDATE:
                    String personsJson = intent.getStringExtra(WebSocketService.EXTRA_PERSONS_JSON);
                    if (personsJson != null) {
                        handlePersonsUpdate(personsJson);
                    }
                    break;

                case WebSocketService.BROADCAST_ERROR:
                    String errorMsg = intent.getStringExtra(WebSocketService.EXTRA_ERROR_MSG);
                    if (errorMsg != null) {
                        runOnUiThread(() ->
                                Toast.makeText(MainActivity.this,
                                        "Error: " + errorMsg, Toast.LENGTH_SHORT).show());
                    }
                    break;
            }
        }
    };

    private void handlePersonsUpdate(String personsJson) {
        runOnUiThread(() -> {
            try {
                JSONArray arr = new JSONArray(personsJson);
                for (int i = 0; i < arr.length(); i++) {
                    JSONObject obj = arr.getJSONObject(i);
                    PostureAlert p = new PostureAlert();
                    p.personId = obj.optInt("person_id", 0);
                    p.statusCode = obj.optString("status_code", "nd");
                    p.statusLabel = obj.optString("status_label", "");
                    p.cpi = obj.optDouble("cpi", 0);
                    p.lumbar = obj.optDouble("lumbar", 0);
                    p.curvature = obj.optDouble("curvature", 0);
                    p.badTime = obj.optDouble("bad_time", 0);
                    p.confidence = obj.optDouble("confidence", 0);
                    p.timestamp = System.currentTimeMillis();
                    adapter.updatePerson(p);
                }
                if (adapter.hasPersons()) {
                    emptyStateText.setVisibility(View.GONE);
                }
            } catch (JSONException ignored) {}
        });
    }

    private void handleAlert(JSONObject alertJson) {
        runOnUiThread(() -> {
            try {
                PostureAlert alert = new PostureAlert();
                alert.personId = alertJson.optInt("person_id", 0);
                alert.statusCode = alertJson.optString("status_code", "nd");
                alert.statusLabel = alertJson.optString("status_label", "");
                alert.cpi = alertJson.optDouble("cpi", 0);
                alert.lumbar = alertJson.optDouble("lumbar", 0);
                alert.curvature = alertJson.optDouble("curvature", 0);
                alert.badTime = alertJson.optDouble("bad_time", 0);
                alert.confidence = alertJson.optDouble("confidence", 0);
                alert.timestamp = System.currentTimeMillis();

                adapter.updatePerson(alert);
                emptyStateText.setVisibility(View.GONE);

                // Show notification if alert
                if (alert.isAlert()) {
                    notificationHelper.showAlertNotification(
                            alert.personId, alert.statusLabel, alert.cpi, alert.badTime);
                }
            } catch (Exception e) {
                // Ignore parse errors
            }
        });
    }

    // ── Lifecycle ─────────────────────────────────────────────────────

    @Override
    protected void onResume() {
        super.onResume();
        // Register broadcast receiver for WebSocket events
        IntentFilter filter = new IntentFilter();
        filter.addAction(WebSocketService.BROADCAST_CONNECTED);
        filter.addAction(WebSocketService.BROADCAST_DISCONNECTED);
        filter.addAction(WebSocketService.BROADCAST_ALERT);
        filter.addAction(WebSocketService.BROADCAST_RESOLUTION);
        filter.addAction(WebSocketService.BROADCAST_PERSON_LEFT);
        filter.addAction(WebSocketService.BROADCAST_PERSONS_UPDATE);
        filter.addAction(WebSocketService.BROADCAST_ERROR);
        broadcastManager.registerReceiver(wsReceiver, filter);

        // If service is running and connected, update UI state
        if (hasSavedPairing()) {
            disconnectButton.setVisibility(View.VISIBLE);
            scanQrButton.setVisibility(View.GONE);
            String sid = getSavedSid();
            if (sid != null) {
                sessionIdText.setText("Sesión: " + sid.substring(0, Math.min(8, sid.length())) + "...");
                sessionIdText.setVisibility(View.VISIBLE);
            }
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        broadcastManager.unregisterReceiver(wsReceiver);
    }

    // NOTE: onDestroy does NOT stop the WebSocketService.
    // The foreground service keeps the WebSocket connection alive
    // even when the app is closed or in the background.

    // ── UI State ──────────────────────────────────────────────────────

    private void updateDisconnected() {
        connectionStatus.setText(R.string.status_disconnected);
        statusDot.setTextColor(0xFF94a3b8);
    }

    private void updateConnecting() {
        connectionStatus.setText(R.string.status_connecting);
        statusDot.setTextColor(0xFFf59e0b);
    }

    private void updateConnected() {
        connectionStatus.setText(R.string.status_connected);
        statusDot.setTextColor(0xFF22c55e);
        progressBar.setVisibility(View.GONE);
        disconnectButton.setVisibility(View.VISIBLE);
    }
}
