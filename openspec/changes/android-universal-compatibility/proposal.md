# Proposal: Android Universal Compatibility

## Intent

The app currently fails to install or run reliably on modern Android 15 devices (like OPPO Reno 11 5G / ColorOS 16.0.2). We need to upgrade the target SDK to 35, enforce proper APK signing, and implement modern background execution requirements (Foreground Services) to ensure the WebSocket connection stays alive without being killed by aggressive battery managers.

## Scope

### In Scope
- Upgrade `targetSdk` and `compileSdk` to 35.
- Add V2/V3 Release APK Signing configuration in `build.gradle`.
- Implement a Foreground Service (`dataSync` type) for WebSocket communication.
- Add runtime permissions handling for Notifications (Android 13+), Camera, and Foreground Service.
- Add Network Security Config for explicit WebSocket security.
- Add user-prompt flows for ColorOS battery optimization (`REQUEST_IGNORE_BATTERY_OPTIMIZATIONS`).

### Out of Scope
- Rewriting the UI layer or changing the core UX.
- Migrating from Java to Kotlin (Java will be maintained).
- Refactoring the entire WebSocket protocol (only the Service wrapper is added).

## Capabilities

### New Capabilities
- `background-websocket`: Foreground service implementation for persistent socket connections.
- `permissions-flow`: Unified runtime permissions and battery optimization prompts.

### Modified Capabilities
- None

## Approach

We will create a `WebSocketService` that extends `Service` and registers as a foreground service with a persistent notification. `WsClient` will be managed inside this service. We will update `build.gradle` to API 35 and add a `signingConfigs` block for release. We will add all necessary permissions to `AndroidManifest.xml` (`FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_DATA_SYNC`, `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS`) and trigger the permission requests in `MainActivity` before establishing the connection.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `mobile-android/app/build.gradle` | Modified | SDK bump, release signing |
| `mobile-android/app/src/main/AndroidManifest.xml` | Modified | New permissions, service declaration |
| `mobile-android/app/src/main/java/com/posturemonitor/ws/` | New/Modified | `WebSocketService.java` creation |
| `mobile-android/app/src/main/java/com/posturemonitor/MainActivity.java` | Modified | Add permission flows |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| ColorOS killing the background service | High | Prompt user for battery unrestricted mode. |
| Foreground service crashing on launch | Medium | Ensure notification channel exists before starting service. |

## Rollback Plan

Revert `build.gradle` to `targetSdk 34`, remove the `<service>` from `AndroidManifest.xml`, and switch `WsClient` instantiation back to `MainActivity` via standard git revert.

## Dependencies

- None

## Success Criteria

- [ ] App installs successfully on OPPO Reno 11 5G / Android 15.
- [ ] WebSocket stays connected when the app is backgrounded.
- [ ] Release APK is properly signed.