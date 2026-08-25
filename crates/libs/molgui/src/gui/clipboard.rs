/// Text-only clipboard bridge — replaces egui-winit's `clipboard` feature (which pulls arboard's
/// `image-data` feature → image crate → moxcms → fearless_simd = ~28 MB .rlib).
/// We use arboard with `default-features = false` (text-only, 18 deps, no image chain).
///
/// Usage in the event loop:
///   1. Before `ctx.run()`: call `inject_paste_if_needed(&mut raw_input, &clipboard, &event)`
///      to translate Ctrl+V into `egui::Event::Paste(text)`.
///   2. After `ctx.run()`: call `handle_output_commands(&full_output.platform_output, &clipboard)`
///      to write `OutputCommand::CopyText` to the OS clipboard.
///
/// The Cut/Copy/Paste key detection mirrors egui-winit 0.34 `is_cut_command`/`is_copy_command`/`is_paste_command`
/// (egui-winit-0.34.3/src/lib.rs:1305-1321).

use arboard::Clipboard as ArboardClipboard;
use egui::{Event, Key, Modifiers, OutputCommand, PlatformOutput};

/// Text-only clipboard wrapper. arboard with default-features=false handles text only.
pub struct Clipboard {
    inner: ArboardClipboard,
}

impl Clipboard {
    pub fn new() -> Self {
        Self { inner: ArboardClipboard::new().expect("arboard: cannot access system clipboard") }
    }

    /// Read text from clipboard. Returns None if empty or not text.
    pub fn get_text(&mut self) -> Option<String> {
        self.inner.get_text().ok()
    }

    /// Write text to clipboard.
    pub fn set_text(&mut self, text: String) {
        // arboard set_text takes &str; we ignore errors (clipboard may be unavailable in headless)
        if let Err(e) = self.inner.set_text(&text) {
            eprintln!("Clipboard::set_text failed: {e}");
        }
    }
}

// --- Key detection (mirrors egui-winit 0.34 lib.rs:1305-1321) ---

fn is_cut_command(modifiers: Modifiers, key: Key) -> bool {
    key == Key::Cut || (modifiers.command && key == Key::X)
        || (cfg!(target_os = "windows") && modifiers.shift && key == Key::Delete)
}
fn is_copy_command(modifiers: Modifiers, key: Key) -> bool {
    key == Key::Copy || (modifiers.command && key == Key::C)
        || (cfg!(target_os = "windows") && modifiers.ctrl && key == Key::Insert)
}
fn is_paste_command(modifiers: Modifiers, key: Key) -> bool {
    key == Key::Paste || (modifiers.command && key == Key::V)
        || (cfg!(target_os = "windows") && modifiers.shift && key == Key::Insert)
}

/// Inject `Event::Paste(text)` into raw_input.events when Ctrl+V / Cmd+V is detected.
/// Call this BEFORE `ctx.run(raw_input, ...)` — but AFTER `egui_state.take_egui_input(&window)`,
/// since take_egui_input populates `raw_input.modifiers` and key events.
///
/// Returns true if a paste was injected (caller may want to consume the key event).
pub fn inject_paste_if_needed(events: &mut Vec<Event>, modifiers: Modifiers, clipboard: &mut Clipboard) -> bool {
    // Check if any key event in this frame is a paste command
    let has_paste_key = events.iter().any(|e| match e {
        Event::Key { key, pressed: true, .. } => is_paste_command(modifiers, *key),
        _ => false,
    });
    if !has_paste_key { return false; }
    // Read clipboard and inject Paste event
    if let Some(text) = clipboard.get_text() {
        let text = text.replace("\r\n", "\n");
        if !text.is_empty() {
            events.push(Event::Paste(text));
            return true;
        }
    }
    false
}

/// Also inject Cut/Copy events if detected (egui-winit does this with clipboard feature;
/// without it, these events are not generated, so text widgets won't know to copy).
pub fn inject_cut_copy_if_needed(events: &mut Vec<Event>, modifiers: Modifiers) {
    // We need to find Key events for Cut/Copy and append the corresponding Event::Cut/Copy
    // (egui text widgets listen for Event::Copy/Cut, not Event::Key with Key::C)
    let mut extra = Vec::new();
    for e in events.iter() {
        if let Event::Key { key, pressed: true, .. } = e {
            if is_cut_command(modifiers, *key) { extra.push(Event::Cut); }
            else if is_copy_command(modifiers, *key) { extra.push(Event::Copy); }
        }
    }
    events.extend(extra);
}

/// Handle `PlatformOutput::commands` — write CopyText to clipboard.
/// Call this AFTER `ctx.run()` with `full_output.platform_output`.
pub fn handle_output_commands(platform_output: &PlatformOutput, clipboard: &mut Clipboard) {
    for cmd in &platform_output.commands {
        match cmd {
            OutputCommand::CopyText(text) => clipboard.set_text(text.clone()),
            OutputCommand::CopyImage(_) => {} // text-only clipboard; image copy not supported
            OutputCommand::OpenUrl(_) => {}   // handled by egui-winit's `links` feature
        }
    }
}
