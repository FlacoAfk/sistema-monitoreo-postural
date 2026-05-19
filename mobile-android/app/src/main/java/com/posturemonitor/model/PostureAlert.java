package com.posturemonitor.model;

/**
 * Datos de postura por persona recibidos del WebSocket.
 */
public class PostureAlert {

    public int personId;
    public String statusCode;   // "ok", "warn", "crit", "nd"
    public String statusLabel;  // "CORRECTO", "ALERTA LEVE", "ALERTA CRÍTICA"
    public double cpi;
    public double lumbar;
    public double curvature;
    public double badTime;
    public double confidence;
    public long timestamp;

    public PostureAlert() {}

    public String getBadgeColor() {
        switch (statusCode) {
            case "ok":   return "#22c55e";
            case "warn": return "#f59e0b";
            case "crit": return "#ef4444";
            case "nd":   return "#94a3b8";
            default:     return "#94a3b8";
        }
    }

    public String getBadgeText() {
        switch (statusCode) {
            case "ok":   return "✓ Correcto";
            case "warn": return "⚠ Leve";
            case "crit": return "✗ Crítico";
            case "nd":   return "— No detectado";
            default:     return "— Sin datos";
        }
    }

    public boolean isAlert() {
        return "warn".equals(statusCode) || "crit".equals(statusCode);
    }
}
