# PostureMonitor — App Android (APK Nativa)

App nativa Android para recibir alertas de postura en tiempo real desde el dashboard de monitoreo postural.

## Características

- **Escaneo QR nativo** con CameraX + ZXing
- **Conexión WebSocket** en tiempo real con reconexión automática
- **Notificaciones del sistema** con vibración al detectar mala postura
- **Datos por persona** — CPI, ángulo lumbar, curvatura, tiempo en mala postura, confianza
- **Diseño responsive** — se adapta a cualquier pantalla
- **Modo pantalla completa** — sin barra de navegación

## Cómo construir el APK

### Opción A: Android Studio (Recomendada)

1. **Instalá Android Studio**: https://developer.android.com/studio

2. **Abrí el proyecto**:
   - Android Studio → File → Open → Seleccioná la carpeta `mobile-android/`

3. **Esperá a que sincronice Gradle** (descarga dependencias automáticamente)

4. **Construí el APK**:
   - Build → Build Bundle(s) / APK(s) → Build APK(s)
   - O atajo: `Ctrl + Shift + F9`

5. **El APK queda en**:
   ```
   mobile-android\app\build\outputs\apk\debug\app-debug.apk
   ```

6. **Instalalo en tu celular**:
   - Transferí el APK al celular
   - Activá "Instalar apps de fuentes desconocidas" en Ajustes
   - Tocá el APK para instalar

### Opción B: Línea de comandos (si tenés Android SDK)

```bash
cd "C:\Users\elkaw\Desktop\Modelos entrenados\mobile-android"
gradlew.bat assembleDebug
```

## Cómo usar la app

1. **Iniciá el dashboard** en tu PC:
   ```powershell
   $env:POSTURE_WS_ENABLED="true"
   python -m src.ui.app
   ```

2. **Abrí la app** en tu celular

3. **Tocá "Escanear QR"** y apuntá al código QR del dashboard

4. **Listo** — la app muestra datos por persona y envía notificaciones

## Estructura del proyecto

```
mobile-android/
├── app/
│   ├── src/main/
│   │   ├── AndroidManifest.xml
│   │   ├── java/com/posturemonitor/
│   │   │   ├── MainActivity.java          # Pantalla principal
│   │   │   ├── model/
│   │   │   │   └── PostureAlert.java       # Modelo de datos
│   │   │   ├── ui/
│   │   │   │   ├── QrScannerActivity.java  # Escaneo QR nativo
│   │   │   │   ├── PersonCardAdapter.java  # Tarjetas por persona
│   │   │   │   └── NotificationHelper.java # Notificaciones + vibración
│   │   │   └── ws/
│   │   │       └── WsClient.java           # WebSocket con reconexión
│   │   └── res/                            # Layouts, colores, temas
│   └── build.gradle                        # Dependencias
└── build.gradle                            # Configuración raíz
```

## Requisitos

- **Android 7.0+** (API 24)
- **Cámara** (para escanear QR)
- **WiFi** (misma red que el dashboard)

## Troubleshooting

| Problema | Solución |
|----------|----------|
| La app no escanea el QR | Verificá permisos de cámara en Ajustes |
| No se conecta | Dashboard debe correr con `POSTURE_WS_ENABLED=true` |
| Se desconecta | PC y celular deben estar en la misma red WiFi |
| No llegan notificaciones | Activá notificaciones en Ajustes → Apps → PostureMonitor |
