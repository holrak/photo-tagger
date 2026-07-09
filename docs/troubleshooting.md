---
icon: lucide/wrench
---

# Troubleshooting

The three failures below account for almost every run that will not start. Each entry shows the
symptom first, then the fix. When in doubt, run the built-in pre-flight check; it tests exactly the
two external pieces (ExifTool and the model server) that these failures come from:

```bash
photo-tagger doctor
```

## "exiftool not found"

**Symptom:** the run (or `doctor`) fails immediately with a message that `exiftool` was not found.

photo-tagger shells out to the [ExifTool](https://exiftool.org/) binary for all metadata reads and
writes, and it does not bundle it. Install it with your system package manager:

```bash
brew install exiftool                  # macOS (Homebrew)
apt install libimage-exiftool-perl    # Debian / Ubuntu
```

or download a build for your platform from [exiftool.org](https://exiftool.org/). See
[Installation](getting-started/installation.md#system-requirements) for more package managers.

After installing, the binary must be discoverable: either on your `PATH` (check with
`exiftool -ver`), or pointed at explicitly. If it lives somewhere unusual, set the `exiftool_path`
key in your [config file](getting-started/configuration.md#config-file) or the
`PHOTO_TAGGER_EXIFTOOL` environment variable (the env var wins) to the full path of the binary.

!!! note

    The desktop GUI, when launched from Finder on macOS, automatically inherits the `PATH` your login
    shell would set, so a Homebrew or MacPorts ExifTool is found even though Finder gives apps a minimal
    `PATH`.

## Connection refused / could not list models

**Symptom:** the run fails with a connection error, or `doctor` reports it could not reach the
provider or list its models.

photo-tagger needs a running model server. Check, in order:

1. **Is the server actually running?** Start Ollama, LM Studio's server, or `llama-server` first.
    The `openai` provider talks to a hosted API, so there check your network and API key instead.

2. **Is it on the port photo-tagger expects?** Each provider has a default base URL:

    | Provider  | Default base URL            |
    | --------- | --------------------------- |
    | Ollama    | `http://localhost:11434/v1` |
    | LM Studio | `http://localhost:1234/v1`  |
    | llama.cpp | `http://localhost:8080/v1`  |
    | OpenAI    | `https://api.openai.com/v1` |

3. **Overriding the URL?** If your server listens elsewhere (a different port, a remote host), pass
    `-u/--url` or set the provider's env var (`OLLAMA_BASE_URL`, `LM_STUDIO_BASE_URL`,
    `LLAMA_CPP_BASE_URL`, or `OPENAI_BASE_URL`).

Then confirm the fix with `photo-tagger doctor --provider <name>`.

## "Model 'X' not available"

**Symptom:** the provider is reachable, but the run stops with a message that the requested model is
not available on it.

photo-tagger validates the `-m/--model` identifier against what the server actually serves before
processing any photo, so a typo or a model that was never pulled fails fast. To fix it:

- **List what the server serves.** `photo-tagger doctor` prints the available models when the check
    fails; in the GUI, press the **Refresh** button next to the model dropdown.
- **Pull or load the model first.** With Ollama, `ollama pull <model>`; with LM Studio, download the
    model in its UI and load it into the server; with llama.cpp, start `llama-server` with the model
    file.
- **Match the id exactly.** The identifier must match the server's spelling, including any namespace
    or tag (for example `qwen/qwen3-vl-30b` or `llava:34b`).
