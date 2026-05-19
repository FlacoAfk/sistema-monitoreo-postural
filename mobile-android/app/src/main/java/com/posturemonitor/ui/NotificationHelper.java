package com.posturemonitor.ui;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.os.VibratorManager;

import androidx.core.app.NotificationCompat;

import com.posturemonitor.MainActivity;
import com.posturemonitor.R;

/**
 * Helper para mostrar notificaciones del sistema y vibrar el dispositivo.
 */
public class NotificationHelper {

    private static final String CHANNEL_ID = "posture_alerts";
    private static final String CHANNEL_NAME = "Alertas Posturales";
    private static final int NOTIFICATION_ID_BASE = 1000;

    private final Context context;
    private final NotificationManager notificationManager;
    private final Vibrator vibrator;

    public NotificationHelper(Context context) {
        this.context = context;
        this.notificationManager = (NotificationManager) context.getSystemService(Context.NOTIFICATION_SERVICE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            vibrator = ((VibratorManager) context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE)).getDefaultVibrator();
        } else {
            vibrator = (Vibrator) context.getSystemService(Context.VIBRATOR_SERVICE);
        }
        createChannel();
    }

    private void createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    CHANNEL_NAME,
                    NotificationManager.IMPORTANCE_HIGH
            );
            channel.setDescription("Notificaciones de mala postura detectada");
            channel.enableVibration(true);
            channel.setVibrationPattern(new long[]{0, 200, 100, 200, 400});
            notificationManager.createNotificationChannel(channel);
        }
    }

    public void showAlertNotification(int personId, String statusLabel, double cpi, double badTime) {
        int notifId = NOTIFICATION_ID_BASE + personId;

        Intent intent = new Intent(context, MainActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        PendingIntent pendingIntent = PendingIntent.getActivity(
                context, 0, intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
        );

        String title = "⚠ Persona " + (personId + 1) + ": " + statusLabel;
        String text = String.format("CPI: %.0f | Tiempo: %.0fs", cpi, badTime);

        NotificationCompat.Builder builder = new NotificationCompat.Builder(context, CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_launcher_foreground)
                .setContentTitle(title)
                .setContentText(text)
                .setStyle(new NotificationCompat.BigTextStyle().bigText(text))
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setAutoCancel(true)
                .setContentIntent(pendingIntent)
                .setVibrate(new long[]{0, 200, 100, 200, 400});

        notificationManager.notify(notifId, builder.build());
        vibrate();
    }

    public void clearNotification(int personId) {
        notificationManager.cancel(NOTIFICATION_ID_BASE + personId);
    }

    public void vibrate() {
        if (vibrator != null && vibrator.hasVibrator()) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createWaveform(
                        new long[]{0, 200, 100, 200, 400}, -1));
            } else {
                vibrator.vibrate(new long[]{0, 200, 100, 200, 400}, -1);
            }
        }
    }
}
