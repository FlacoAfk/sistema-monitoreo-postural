# Delta for System

## ADDED Requirements

### Requirement: Backend Model Loading Path Fix

The system MUST load pose detection models using an absolute path derived from `REPO_ROOT` to ensure compatibility across different execution environments.

#### Scenario: Backend initializes models
- GIVEN the backend is starting in an arbitrary directory
- WHEN the model loading routine is executed
- THEN the system MUST resolve model paths relative to `REPO_ROOT`
- AND the models MUST load successfully without `FileNotFound` errors.

### Requirement: Status Code Mapping Fix

The backend MUST send status codes as normalized string values (`"ok"`, `"warn"`, `"crit"`, `"nd"`) to the frontend via WebSocket, rather than exposing internal enum names.

#### Scenario: Status emission via WebSocket
- GIVEN a posture status is evaluated by the backend
- WHEN the status is emitted to connected WebSocket clients
- THEN the payload's `status` field MUST be exactly one of `"ok"`, `"warn"`, `"crit"`, or `"nd"`.

### Requirement: Immediate AlertRouter Emission

The frontend `AlertRouter` MUST emit an alert immediately upon the first entry into an `ALERTA_LEVE` (warn) or `ALERTA_CRITICA` (crit) state, instead of waiting for milestone thresholds.

#### Scenario: First entry into critical posture
- GIVEN the user is in an `"ok"` posture state
- WHEN the backend sends a `"crit"` status
- THEN the `AlertRouter` MUST immediately emit a critical alert
- AND subsequent alerts SHOULD follow the standard milestone configuration.

### Requirement: Constrained Model Selector

The frontend model selector MUST only display the 4 best-performing models, explicitly excluding `yolov11m-pose` from the available options due to user feedback.

#### Scenario: User selects a model
- GIVEN the user opens the model selection dropdown
- WHEN the list of available models is rendered
- THEN the list MUST contain at most 4 models
- AND `yolov11m-pose` MUST NOT be in the list.

### Requirement: Resilient WsClient Connection

The frontend `WsClient` MUST implement connection reliability measures, including detailed logging of connection state changes and robust automatic reconnection handling.

#### Scenario: WebSocket connection drop
- GIVEN the frontend is connected to the backend via WebSocket
- WHEN the connection is unexpectedly dropped
- THEN the `WsClient` MUST log the disconnection event explicitly
- AND the `WsClient` MUST automatically attempt to reconnect with backoff until successful.
