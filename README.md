# ChangeScope : AI-Powered Satellite Image Change Detection


We propose an AI-powered edge-based satellite change-screening system that locally processes SpaceNet imagery on Arduino UNO Q, detects changes using heatmaps and overlays, and generates area-wise reports over time. Significant anomalies trigger email alerts with inspection reports, while local AI processing ensures privacy, security, and low latency.

<img width="750" height="411" alt="image" src="https://github.com/user-attachments/assets/6f2eb5c1-37b4-4c7d-bc5c-fc22c63d91b5" />

ChangeScope is a Python-based application for comparing satellite or other Earth-observation images captured at different times. It helps users inspect visual differences, review change maps and timeline results, and turn technical detection output into understandable analysis.

> **Repository layout matters:** This repository primarily contains the laptop/desktop-runnable version of ChangeScope. A separate folder in the repository contains the version adapted and optimized for the **Arduino UNO Q**. Use the main project for development and general-purpose computer testing; use the Arduino UNO Q-specific folder and its instructions when deploying to the board. The two versions share the same project goal, but their dependency and runtime choices may differ.

## Contents

- [Project goals](#project-goals)
- [Key features](#key-features)
- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Run the laptop/desktop version](#run-the-laptopdesktop-version)
- [Arduino UNO Q version](#arduino-uno-q-version)
- [Configuration](#configuration)
- [Using the dashboard](#using-the-dashboard)
- [Models and AI interpretation](#models-and-ai-interpretation)
- [Email alerts](#email-alerts)
- [Troubleshooting](#troubleshooting)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [Contributing](#contributing)

## Project goals

ChangeScope is designed to make multi-date image review more accessible by combining image comparison, visual results, a web dashboard, and optional AI-assisted interpretation.

Potential use cases include:

- Monitoring visible land-cover or land-use changes.
- Reviewing construction, vegetation, water, or other surface changes where the input imagery supports it.
- Comparing images from different dates for an area of interest.
- Producing visual evidence and summaries for human review.

Results depend on image quality, resolution, viewpoint, lighting, season, and alignment. A visual difference is not automatically proof of a real-world event.

## Key features

Feature availability can vary slightly between the laptop version and the Arduino UNO Q package.

- **Image comparison:** compare images and inspect detected visual differences.
- **Timeline analysis:** review a sequence of images captured at multiple time points, where supported by the selected workflow.
- **Change maps and visualizations:** present results in the dashboard, including charts or heatmap-style views provided by the application.
- **Web interface:** use a browser-based dashboard backed by a Python/Flask application.
- **ONNX inference:** the UNO Q package uses ONNX Runtime and an ONNX model to avoid requiring the full PyTorch stack on the board.
- **Gemma AI interpretation:** the project can use Gemma to produce readable summaries from analysis context. Follow the configuration and implementation in the version you are running; local Gemma inference requires a compatible model/runtime to be installed and configured.
- **Reports:** PDF/report generation is available where the relevant optional dependencies and routes are included.
- **Email notifications:** email alerts can be enabled through SMTP configuration.
- **Modular backend:** detection, timeline, AI, reporting, notifications, and web routes are organized into separate modules.

## How it works

1. **Provide imagery.** Upload image files for the same area from different dates. The current workflow should be treated as file-based unless you have separately integrated a live imagery provider or satellite feed.
2. **Validate and prepare inputs.** The application checks and prepares images for processing.
3. **Run change detection.** The detection pipeline extracts image features and compares inputs to estimate visual differences.
4. **Review temporal results.** Where timeline analysis is enabled, review the sequence and comparisons across dates.
5. **Interpret results.** The dashboard presents visual output, and the configured Gemma integration may generate a text explanation.
6. **Export or notify.** Use available reporting features and configure email alerts if required.

## Architecture

<img width="4560" height="2565" alt="Start" src="https://github.com/user-attachments/assets/0a90d8a3-d1ab-4845-a29b-5157848399a2" />


### Main laptop/desktop application

- **Frontend:** HTML, CSS, JavaScript, and browser-side visualization components.
- **Web server:** Python and Flask.
- **Image processing:** Python imaging libraries such as OpenCV and Pillow, depending on the installed version.
- **Detection:** the configured feature-extraction/change-detection pipeline.
- **AI interpretation:** Gemma integration, depending on the selected runtime and configuration.
- **Outputs:** dashboard visualizations, timeline results, reports, and optional email notifications.

### Arduino UNO Q deployment

The dedicated UNO Q package is adapted for the board's Linux environment and constrained resources. Its documented dependency set uses **ONNX Runtime** for model inference and **OpenCV** for image processing rather than depending on PyTorch, torchvision, and SciPy for the core detection path. The package includes a modular Flask application and its own requirements/instructions.

The UNO Q detection model runs on the board's CPU in the described deployment. This does not mean every optional component automatically runs locally: Gemma's location and execution depend on how its local runtime/model has been installed and configured.

### Important satellite integration note

ChangeScope analyzes imagery supplied to it. Listing satellite imagery as the source in an architecture diagram describes the upstream data workflow; it does **not** imply that this repository directly controls a satellite, receives a live satellite downlink, automatically downloads imagery, or performs full GIS/georeferencing correction unless those capabilities have been added separately.

## Repository layout

The repository is organized around two ways to run the project:

```text
ChangeScope/
├── README.md                  # This document
├── app.py / run.py            # Entry point(s), depending on the version
├── requirements.txt           # Main laptop/desktop dependencies, if present
├── templates/                 # Web dashboard templates
├── static/                    # Frontend assets, if present
├── models/                    # Model files, if supplied separately
├── changescope/               # Modular application package in the UNO Q port
└── <UNO_Q_FOLDER>/            # Dedicated Arduino UNO Q version in this repository
    ├── README_UNO_Q.md        # Board-specific instructions, if present
    ├── requirements-uno-q.txt
    ├── run.py
    └── changescope/
```

The exact filenames can differ between repository revisions. Keep the main laptop/desktop application separate from the UNO Q-adapted package, and follow the README and dependency file inside the specific folder you intend to run. Do not assume the two folders use identical entry points, requirements, or model paths.

## Requirements

### Laptop/desktop version

- Python 3.10 or a version supported by the repository's dependencies.
- `pip` and Python virtual-environment support.
- A compatible model file in the path expected by the configuration.
- Optional credentials and services for AI interpretation and SMTP email.

Use the project's own `requirements.txt` if present. Do not install the UNO Q requirements into the laptop environment unless you intentionally want that dependency set.

### Arduino UNO Q version

- Arduino UNO Q with its supported Linux environment and sufficient free storage/RAM.
- Python 3 and virtual-environment support.
- The UNO Q folder's `requirements-uno-q.txt`.
- The ONNX model at the configured path.
- Network access only for features that need network services; locally configured inference can have different requirements.

Consult `README_UNO_Q.md` inside the board-specific package for its exact setup instructions.

## Run the laptop/desktop version

From the repository root, first inspect the actual entry point and dependency file in your checkout. The common setup pattern is:

### 1. Create and activate a virtual environment

**Windows PowerShell**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**
```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the main dependencies

If the repository root contains `requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the root uses a different dependency file, install the file provided for that version instead.

### 3. Check model files and configuration

Make sure the model file expected by the application exists. Do not assume that a large model binary is included in Git; check the repository's model directory and project instructions.

### 4. Start the server

Use the entry point present in the main project. Common examples are:

```bash
python app.py
```

or:

```bash
python run.py
```

These are alternatives, not commands that must both be run. If the project documents a different command, follow that instruction.

### 5. Open the dashboard

Open the local URL printed by the Flask server, commonly:

```text
http://127.0.0.1:5000
```

To access the dashboard from another device on the same trusted network, the server must be configured to listen on the appropriate interface and firewall rules must permit access. Do not expose a development Flask server directly to the public internet.

## Arduino UNO Q version

The UNO Q-specific implementation is maintained in its dedicated folder; it is not the default target for every file in this repository.

1. Change into the UNO Q folder.
2. Read its `README_UNO_Q.md`.
3. Create and activate a Python virtual environment on the board.
4. Install the board-specific dependencies.
5. Place or link the ONNX model at the path expected by that package.
6. Configure any optional services.
7. Start the application using the documented entry point.

A typical setup pattern for the UNO Q package is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-uno-q.txt
python run.py
```

Run these commands **from the UNO Q package directory** and only after confirming its files are present. If the board-specific README provides a different command, use that version's instructions.

For a service deployment, the package may include a systemd service unit. Review and edit its paths and user settings for your board before enabling it; do not install a service file blindly.

## Configuration

Configuration names can differ by version. Check the configuration module and example environment file in the exact version you are running. Keep secrets out of source control.

A `.env` file may be used for optional integrations. For the versions using the settings below:

```dotenv
# Optional Gemma/Google AI integration, if this version uses an API
GEMINI_API_KEY=your_api_key_here
GEMINI_KEY=your_api_key_here
GEMINI_MODEL=gemma-4-26b-a4b-it

# Optional upload limit, in megabytes
MAX_CONTENT_LENGTH_MB=32
```

**Important:** These API-key variables apply to an API-based AI client. If your deployment uses a genuinely local Gemma runtime, configure that runtime and its local model path as required by its implementation instead; an API key should not be presented as a requirement for local inference.

Never commit real API keys, SMTP passwords, or other secrets. Add `.env` to `.gitignore` and share a redacted `.env.example` instead.

Other settings, such as change thresholds, minimum region size, model paths, email recipients, and SMTP server details, should be confirmed from the configuration file for the version being used rather than assumed to be identical across both targets.

## Using the dashboard

1. Start the server and open the dashboard URL.
2. Choose the available single-comparison or timeline workflow.
3. Upload the required image file(s). For timeline analysis, use images representing the same area and provide dates in the format requested by the interface.
4. Run the analysis and wait for processing to finish.
5. Inspect the comparison output, change visualization, charts, and any available AI summary.
6. Export a report or configure notifications if those features are enabled.

For meaningful comparisons, use images with similar coverage, resolution, orientation, and acquisition conditions where possible. Large differences caused by clouds, shadows, seasonal variation, camera angle, or image misalignment can be mistaken for meaningful change.

## Models and AI interpretation

### Change-detection model

The UNO Q package is designed to use an ONNX model, referenced in the known configuration as:

```text
models/resnet18_encoder.onnx
```

Verify the actual path in your checkout. The model may need to be obtained separately if it is not included in the repository. Use only a model file compatible with the preprocessing and output expectations of the code.

### Gemma interpretation

Gemma is a separate AI interpretation component; it is not the same thing as the ONNX change-detection model.

- The detection model produces the computer-vision results.
- Gemma can interpret the supplied analysis context and produce a natural-language summary.
- Local Gemma inference requires the model and a compatible local inference runtime to be installed and configured.
- An API-based Gemma client instead requires network connectivity and valid API credentials.

Do not assume the AI summary is available simply because the detection model works. Check the configured AI path and inspect the logs if interpretation fails.

## Email alerts

Email is a separate optional notification component.

To enable it:

1. Inspect the email configuration module for the exact environment variable names expected by your version.
2. Configure the SMTP host, port, sender account, credentials, recipient list, and TLS/SSL mode as required.
3. Keep credentials in environment variables or a local `.env` file, not in source code.
4. Test delivery with a non-sensitive test message before relying on alerts.

Email notifications are only sent if the feature is implemented, configured, and triggered by the application's alert conditions. Email is not required for core image comparison.

## Troubleshooting

### Dashboard does not open

- Confirm the server process is running.
- Read the terminal output for startup errors.
- Confirm the port and host address used by the server.
- Check firewall and network settings if accessing from another device.

### A dependency fails to install

- Confirm the active Python version and virtual environment.
- Use the dependency file for the target you are running.
- On UNO Q, prefer the board-specific requirements rather than installing heavyweight laptop-oriented computer-vision dependencies unnecessarily.

### Model file not found

- Check that the expected ONNX file exists.
- Confirm the configured path and working directory.
- Check file permissions and ensure the model download/copy completed.

### Upload rejected or too large

- Check the application's upload limit. Some versions default to 32 MB through `MAX_CONTENT_LENGTH_MB`.
- Reduce the image size or change the configured limit carefully.
- Consider available memory and storage on the UNO Q before increasing the limit.

### Gemma interpretation fails

- Determine whether the current version uses a local Gemma runtime or an external API.
- For local inference, verify the model, runtime, model path, and available resources.
- For API inference, verify credentials, connectivity, and the configured model identifier.
- Review the server logs for the actual error.

### Email alerts do not arrive

- Verify SMTP settings and recipient addresses.
- Check authentication, TLS/SSL mode, and spam folders.
- Review application logs and confirm the alert condition was actually triggered.

## Limitations and responsible use

- **Uploaded imagery by default:** direct satellite acquisition or live feeds require a separate integration.
- **Visual change is not semantic certainty:** the system highlights image differences; interpretation should be verified by a human.
- **Image alignment matters:** misregistered images can produce false positives.
- **No guaranteed geospatial accuracy:** do not treat displayed pixels or heatmaps as surveyed coordinates unless georeferencing and validation are explicitly implemented.
- **Compute constraints:** large images and local language models can require significant memory, storage, and processing time, particularly on edge hardware.
- **Optional integrations:** AI summaries, reports, and email depend on the relevant modules and configuration being available.

## Contributing

Contributions and bug reports are welcome. Before submitting changes:

1. Identify whether the change targets the main laptop/desktop app, the UNO Q package, or both.
2. Keep platform-specific dependencies and instructions separate.
3. Test the affected workflow and document any new configuration.
4. Never commit credentials, private imagery, or large model files unless the repository's licensing and storage policy explicitly allows them.
5. Include reproduction steps and relevant logs when reporting a bug, with secrets removed.

## License and model terms

Add the project's chosen software license and any third-party model/license notices before distributing the repository. Confirm the redistribution terms for model weights, datasets, and dependencies separately.

---

**Project summary:** ChangeScope combines image-based change detection, temporal review, a browser dashboard, optional Gemma interpretation, reports, and email notifications. The main repository is intended for laptop/desktop development and execution, while a dedicated folder contains the Arduino UNO Q-adapted implementation.
