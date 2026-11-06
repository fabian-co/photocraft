//! Select Subject through the Craft AI layer: when the desktop app was given an AI selection
//! service ([`Services::ai_select`](crate::Services::ai_select)) and the person chose a model for
//! the suite's "Select subject" tool, Select › Subject (and every Select Subject button) asks that
//! model instead of the built-in heuristic. Without a service, or with no model ready, nothing
//! changes: the built-in command runs as before.
//!
//! This crate knows nothing about providers, keys or consent files: the service answers whether
//! the tool can run, asks for consent when the chosen model is online, and runs the job off the
//! interface thread. This module draws the consent prompt, polls the job each frame (Esc cancels)
//! and turns the returned mask into one undoable selection (`select.fromMask`).

use crate::{PhotocraftApp, state::DialogKind};
use base64::Engine as _;
use serde_json::{Map, Value, json};

const MARK: &str = "__aiSelectConsent";
/// What a text-reading model is told to find; models that select by clicking get the hint box.
const PROMPT: &str = "the main subject";

/// Whether the suite's Select Subject tool can run now in this app.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AiSelectStatus {
    /// A model is chosen and this app can run it; `model` is its name for the status line.
    Ready { model: String },
    /// Not usable; `reason` is a sentence for a person.
    Unavailable { reason: String },
}

/// The picture and the hint the model reads.
#[derive(Clone, Debug)]
pub struct AiSelectRequest {
    /// The picture, PNG.
    pub png: Vec<u8>,
    /// Where the subject probably is, `[x0, y0, x1, y1]` as fractions of the picture.
    pub area: Option<[f32; 4]>,
    /// The words for a model that reads them.
    pub prompt: String,
}

/// How long an agreement to send pictures lasts, shortest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AiConsentTerm {
    ThisTime,
    OneHour,
    OneDay,
    UntilTurnedOff,
}

impl AiConsentTerm {
    pub const ALL: [Self; 4] = [Self::ThisTime, Self::OneHour, Self::OneDay, Self::UntilTurnedOff];

    fn id(self) -> &'static str {
        match self {
            Self::ThisTime => "this_time",
            Self::OneHour => "one_hour",
            Self::OneDay => "one_day",
            Self::UntilTurnedOff => "until_revoked",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ThisTime => tl!("Just this time"),
            Self::OneHour => tl!("For one hour"),
            Self::OneDay => tl!("For one day"),
            Self::UntilTurnedOff => tl!("Until I turn it off"),
        }
    }

    fn parse(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.id() == id)
    }
}

/// What the person is asked before a picture leaves this computer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AiConsentAsk {
    /// Where it goes (`https://host`, or the service's name).
    pub destination: String,
    /// What is sent, one line per item ("Picture, 1.2 MB").
    pub items: Vec<String>,
}

/// The answer to [`AiSelect::start`].
pub enum AiSelectStart {
    Started(Box<dyn AiSelectJob>),
    /// Nothing was sent: ask, then call `start` again with the person's answer.
    NeedsConsent(AiConsentAsk),
}

/// One poll of a running selection.
#[derive(Clone, Debug, PartialEq)]
pub enum AiJobPoll {
    /// Still working; percent when the model reports it.
    Running(Option<u8>),
    /// The grayscale mask (PNG, white selected) and whether the model found anything.
    Done {
        mask_png: Vec<u8>,
        matched: bool,
        name: Option<String>,
    },
    Failed(String),
    Cancelled,
}

/// A selection job running away from the interface thread.
pub trait AiSelectJob {
    fn poll(&mut self) -> AiJobPoll;
    fn cancel(&mut self);
}

/// The AI selection service the desktop app installs.
pub trait AiSelect {
    /// Reads no key and sends nothing.
    fn status(&self) -> AiSelectStatus;
    /// Plans the job and, when no consent is needed or `consent` gives it, starts it.
    fn start(&self, request: AiSelectRequest, consent: Option<AiConsentTerm>) -> Result<AiSelectStart, String>;
    /// Draws the service's part of Edit › Preferences › Integrations: which model Select Subject
    /// uses and, for a model that runs on this computer, getting its files. Choices are shared
    /// with the other Craft apps and apply at once, not on OK. Nothing is drawn by default.
    fn preferences_ui(&mut self, _ui: &mut egui::Ui) {}
}

/// The AI block under Edit › Preferences › Integrations, when the app has an AI service.
pub(crate) fn preferences_section(ui: &mut egui::Ui, service: &mut (dyn AiSelect + 'static)) {
    ui.add_space(10.0);
    ui.separator();
    ui.label(egui::RichText::new(tl!("AI models")).strong());
    ui.label(
        egui::RichText::new(tl!("Select › Subject uses the model chosen here. The choice is shared with the other Craft apps and applies at once."))
            .small()
            .weak(),
    );
    ui.add_space(6.0);
    service.preferences_ui(ui);
}

/// A running job, or a request waiting for the consent prompt.
#[derive(Default)]
pub(crate) struct Pending {
    running: Option<Running>,
    /// The request the open consent prompt is about (too big for the dialog's fields).
    awaiting: Option<Awaiting>,
}

struct Running {
    job: Box<dyn AiSelectJob>,
    document: u64,
    mode: Value,
    model: String,
}

struct Awaiting {
    request: AiSelectRequest,
    document: u64,
    mode: Value,
    model: String,
}

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

/// Whether an AI selection is running (for tests and the status line).
pub fn busy(app: &PhotocraftApp) -> bool {
    app.ai_select.running.is_some()
}

/// Takes over `select.subject` when an AI model is ready; `None` lets the built-in command run.
/// Automation keeps the built-in, deterministic command.
pub fn intercept(app: &mut PhotocraftApp, command: &str, params: &Value) -> Option<Result<Value, String>> {
    if command != "select.subject" || app.automation_input || params.get("builtIn").and_then(Value::as_bool) == Some(true) {
        return None;
    }
    let service = app.services.ai_select.as_ref()?;
    let AiSelectStatus::Ready { model } = service.status() else { return None };
    if app.ai_select.running.is_some() {
        return Some(Ok(json!({"pending": true})));
    }
    let all_layers = params.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(false);
    let source = match app.run_command("select.aiSource", json!({"sampleAllLayers": all_layers})) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    let png = match source.get("png").and_then(Value::as_str).map(|s| base64::engine::general_purpose::STANDARD.decode(s)) {
        Some(Ok(bytes)) => bytes,
        _ => return Some(Err(tl!("The picture could not be prepared for the AI model").into())),
    };
    let area = source.get("box").and_then(Value::as_array).and_then(|b| {
        let v: Vec<f32> = b.iter().filter_map(|n| n.as_f64().map(|f| f as f32)).collect();
        (v.len() == 4 && v[0] < v[2] && v[1] < v[3]).then(|| [v[0], v[1], v[2], v[3]])
    });
    let document = source.get("document").and_then(Value::as_u64).unwrap_or(0);
    let mode = params.get("mode").cloned().unwrap_or_else(|| json!("replace"));
    let request = AiSelectRequest { png, area, prompt: PROMPT.into() };
    Some(begin(app, request, document, mode, model, None))
}

fn begin(
    app: &mut PhotocraftApp,
    request: AiSelectRequest,
    document: u64,
    mode: Value,
    model: String,
    consent: Option<AiConsentTerm>,
) -> Result<Value, String> {
    let Some(service) = app.services.ai_select.as_ref() else { return Err(tl!("AI selection is not available").into()) };
    match service.start(request.clone(), consent) {
        Ok(AiSelectStart::Started(job)) => {
            app.ai_select.running = Some(Running { job, document, mode, model: model.clone() });
            app.ui.status = crate::i18n::fmt(tl!("Selecting the subject with {model}… (Esc cancels)"), &[("model", &model)]);
            app.ui.status_error = false;
            Ok(json!({"pending": true, "model": model}))
        }
        Ok(AiSelectStart::NeedsConsent(ask)) => {
            if consent.is_some() {
                return Err(tl!("The AI model still needs your agreement. Please try again.").into());
            }
            app.ai_select.awaiting = Some(Awaiting { request, document, mode, model: model.clone() });
            if let Some(open) = app.ui.dialogs.iter().find(|d| owns(&d.fields)) {
                return Ok(json!({"dialog": open.id, "consentPending": true}));
            }
            let message = crate::i18n::fmt(
                tl!("Select Subject with {model} sends the picture to {destination}."),
                &[("model", &model), ("destination", &ask.destination)],
            );
            let fields = json!({MARK: true, "__label": "Select Subject", "message": message, "items": ask.items, "term": AiConsentTerm::ThisTime.id()});
            let id = app.ui.open_dialog(DialogKind::Command, fields.as_object().cloned().unwrap_or_default());
            Ok(json!({"dialog": id, "consentPending": true}))
        }
        Err(message) => {
            app.ui.status = message.clone();
            app.ui.status_error = true;
            Err(message)
        }
    }
}

pub fn body(ui: &mut egui::Ui, fields: &mut Map<String, Value>) {
    ui.label(fields.get("message").and_then(Value::as_str).unwrap_or_default());
    if let Some(items) = fields.get("items").and_then(Value::as_array) {
        ui.add_space(4.0);
        for item in items.iter().filter_map(Value::as_str) {
            ui.label(format!("• {item}"));
        }
    }
    ui.add_space(8.0);
    ui.label(tl!("Allow it:"));
    let mut term = fields.get("term").and_then(Value::as_str).and_then(AiConsentTerm::parse).unwrap_or(AiConsentTerm::ThisTime);
    for t in AiConsentTerm::ALL {
        ui.radio_value(&mut term, t, t.label());
    }
    fields.insert("term".into(), json!(term.id()));
}

pub fn confirm(app: &mut PhotocraftApp, fields: &Map<String, Value>) -> Result<Value, String> {
    let term = fields.get("term").and_then(Value::as_str).and_then(AiConsentTerm::parse).unwrap_or(AiConsentTerm::ThisTime);
    let Awaiting { request, document, mode, model } = app.ai_select.awaiting.take().ok_or(tl!("Nothing is waiting to be sent"))?;
    if app.session.active().map(|d| d.doc.id.0) != Some(document) {
        return Err(tl!("The document changed. Please choose Select Subject again.").into());
    }
    begin(app, request, document, mode, model, Some(term))
}

/// Polls the running selection; applies the mask when it arrives. Esc cancels.
pub(crate) fn tick(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some(running) = app.ai_select.running.as_mut() else { return };
    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        running.job.cancel();
    }
    match running.job.poll() {
        AiJobPoll::Running(_) => ctx.request_repaint_after(std::time::Duration::from_millis(100)),
        AiJobPoll::Cancelled => {
            app.ai_select.running = None;
            app.ui.status = tl!("Select Subject cancelled").into();
            app.ui.status_error = false;
        }
        AiJobPoll::Failed(message) => {
            app.ai_select.running = None;
            app.ui.status = message;
            app.ui.status_error = true;
        }
        AiJobPoll::Done { mask_png, matched, name } => {
            let Some(Running { document, mode, model, .. }) = app.ai_select.running.take() else { return };
            if !matched {
                app.ui.status = tl!("No subject found — try Object Selection or the Lasso").into();
                app.ui.status_error = false;
                return;
            }
            let params = json!({
                "png": base64::engine::general_purpose::STANDARD.encode(mask_png),
                "document": document,
                "mode": mode,
                "label": "Select Subject",
            });
            match app.run_command("select.fromMask", params) {
                Ok(v) if v.get("changed").and_then(Value::as_bool) == Some(false) => {
                    app.ui.status = tl!("No subject found — try Object Selection or the Lasso").into();
                    app.ui.status_error = false;
                }
                Ok(_) => {
                    app.ui.status = match name {
                        Some(name) => crate::i18n::fmt(tl!("Selected {name} with {model}"), &[("name", &name), ("model", &model)]),
                        None => crate::i18n::fmt(tl!("Subject selected with {model}"), &[("model", &model)]),
                    };
                    app.ui.status_error = false;
                }
                Err(e) => {
                    app.ui.status = e;
                    app.ui.status_error = true;
                }
            }
            ctx.request_repaint();
        }
    }
}

#[cfg(test)]
mod tests;
