# effGen VSCode Extension

> **Experimental.** This extension is a developer-experience preview shipped from
> the repository (`tools/vscode-effgen`). It is not published to the VSCode
> Marketplace and is not covered by effGen's stability guarantees. Build it from
> source (below).

The effGen VSCode extension adds editor support for the effGen framework inside Visual Studio Code.

## Features

| Feature | Description |
|---------|-------------|
| **Prompt-template completion** | Triggers on `LibraryPrompt(`, `effgen.prompts.`, or `%effgen_` — offers all built-in templates with snippet insertion. |
| **Run code lens** | Shows a **▶ Run with effGen** button above any `LibraryPrompt(`, `effgen_chat(`, `%%effgen_agent`, or `%effgen_chat` line. Clicking sends the prompt to the configured effGen server and prints the response to the **effGen** output channel. |
| **Hover docs** | Hover over a quoted template name (e.g. `"coding"`) to see the template's description, category, and usage snippet. |
| **Prompt registry viewer** | Run **effGen: Show Prompt Registry** from the Command Palette (`Ctrl+Shift+P`) to open a webview listing all templates. |

## Requirements

- VS Code ≥ 1.85
- Node.js 18+ (for building the `.vsix` from source)

## Installation

### From source (development)

```bash
cd tools/vscode-effgen
npm install
npm run compile
# To package as .vsix:
# npm install -g @vscode/vsce
# vsce package
```

Then install the generated `.vsix` via **Extensions → Install from VSIX…**

### Pre-built .vsix

Download `vscode-effgen-1.0.1.vsix` from the release page and install via **Extensions → Install from VSIX…**

## Configuration

Open **Settings** (`Ctrl+,`) and search for `effgen`:

| Setting | Default | Description |
|---------|---------|-------------|
| `effgen.serverUrl` | `http://localhost:8080` | URL of your running effGen API server |
| `effgen.defaultModel` | `gpt-5-nano` | Default model id sent to the server for the Run code lens (use any id your server can serve) |
| `effgen.enableCompletion` | `true` | Toggle prompt-template auto-completion |
| `effgen.enableHover` | `true` | Toggle hover documentation |

## Usage

### Prompt-template completion

In any Python file, start typing a `LibraryPrompt(` call and press `Tab`:

```python
LibraryPrompt("cod|")  # ← Tab here completes to "coding" with a snippet
```

### Run code lens

```python
# A ▶ Run with effGen lens appears above this line ↓
result = effgen_chat("Explain gradient descent in one paragraph")
```

Clicking the lens sends the extracted prompt to `effgen.serverUrl/v1/chat/completions` and shows the response in the **effGen** output channel.

### Hover docs

```python
LibraryPrompt("reasoning", ...)
#              ↑ hover here for docs
```

## Development

```bash
# Watch mode (incremental compile)
npm run watch
```
