//! Select Subject and Object Selection through the Craft AI layer: when the desktop app was given
//! an AI selection service ([`Services::ai_select`](crate::Services::ai_select)) and the person
//! chose a model for the suite's "Select subject" tool, Select › Subject (and every Select Subject
//! button) and the Object Selection tool's rectangle and click ask that model instead of the
//! built-in heuristics. Without a service, or with no model ready, nothing changes: the built-in
//! commands run as before.
//!
//! Edit › Generative Fill works the same way with the suite's "Generative fill" tool: a prompt
//! asks what should be in the selection, and the picture that comes back becomes a new layer
//! masked by the selection (`layer.fromAiImage`).
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
    /// Where the subject is, `[x0, y0, x1, y1]` as fractions of the picture.
    pub area: Option<[f32; 4]>,
    /// A click on the object, `[x, y]` as fractions of the picture (used when there is no area).
    pub click: Option<[f32; 2]>,
    /// The words for a model that reads them.
    pub prompt: String,
}

/// What Generative Fill sends: the picture, where to paint, and what to paint there.
#[derive(Clone, Debug)]
pub struct AiFillRequest {
    /// The picture, RGBA PNG.
    pub png: Vec<u8>,
    /// The same size as the picture, RGBA PNG: transparent where the model should paint.
    pub mask: Vec<u8>,
    /// The person's description.
    pub prompt: String,
}

/// What a job is for: a selection or a fill.
#[derive(Clone, Debug)]
enum Request {
    Select(AiSelectRequest),
    Fill(AiFillRequest),
}

/// Which command the AI is answering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Select › Subject: the whole picture, with the built-in subject's bounds as the hint box.
    Subject,
    /// The Object Selection tool: the dragged rectangle, or the object under a click.
    Object,
    /// Edit › Generative Fill: a new layer painted into the selection.
    Fill,
}

impl Kind {
    /// The history step's name.
    fn label(self) -> &'static str {
        match self {
            Self::Subject => "Select Subject",
            Self::Object => "Object Selection",
            Self::Fill => "Generative Fill",
        }
    }

    fn working(self) -> &'static str {
        match self {
            Self::Subject => tl!("Selecting the subject with {model}… (Esc cancels)"),
            Self::Object => tl!("Selecting the object with {model}… (Esc cancels)"),
            Self::Fill => tl!("Generating with {model}… (Esc cancels)"),
        }
    }

    fn sends(self) -> &'static str {
        match self {
            Self::Subject => tl!("Select Subject with {model} sends the picture to {destination}."),
            Self::Object => tl!("Object Selection with {model} sends the picture to {destination}."),
            Self::Fill => tl!("Generative Fill with {model} sends the picture, the selection and your description to {destination}."),
        }
    }

    fn cancelled(self) -> &'static str {
        match self {
            Self::Subject => tl!("Select Subject cancelled"),
            Self::Object => tl!("Object Selection cancelled"),
            Self::Fill => tl!("Generative Fill cancelled"),
        }
    }

    fn nothing(self) -> &'static str {
        match self {
            Self::Subject => tl!("No subject found — try Object Selection or the Lasso"),
            Self::Object => tl!("No object found there — try a tighter rectangle"),
            Self::Fill => tl!("The AI model returned no picture. Try another description."),
        }
    }

    fn done(self) -> &'static str {
        match self {
            Self::Subject => tl!("Subject selected with {model}"),
            Self::Object => tl!("Object selected with {model}"),
            Self::Fill => tl!("Generated with {model}"),
        }
    }

    fn again(self) -> &'static str {
        match self {
            Self::Subject => tl!("The document changed. Please choose Select Subject again."),
            Self::Object => tl!("The document changed. Please select the object again."),
            Self::Fill => tl!("The document changed. Please choose Generative Fill again."),
        }
    }
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

/// One poll of a running job.
#[derive(Clone, Debug, PartialEq)]
pub enum AiJobPoll {
    /// Still working; percent when the model reports it.
    Running(Option<u8>),
    /// The result PNG and whether the model found anything: for a selection the grayscale mask
    /// (white selected), for a fill the picture.
    Done {
        png: Vec<u8>,
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
    /// Whether the suite's Generative Fill tool can run now. Reads no key and sends nothing.
    fn fill_status(&self) -> AiSelectStatus {
        AiSelectStatus::Unavailable { reason: tl!("Generative Fill needs an AI image model").into() }
    }
    /// Like [`start`](Self::start), for Generative Fill.
    fn start_fill(&self, _request: AiFillRequest, _consent: Option<AiConsentTerm>) -> Result<AiSelectStart, String> {
        Err(tl!("Generative Fill needs an AI image model").into())
    }
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
        egui::RichText::new(tl!("PhotoCraft's AI tools use the models chosen here. Choices are shared with the other Craft apps and apply at once."))
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
    kind: Kind,
    document: u64,
    mode: Value,
    model: String,
    /// The new layer's name (Generative Fill).
    name: String,
}

struct Awaiting {
    request: Request,
    kind: Kind,
    document: u64,
    mode: Value,
    model: String,
    name: String,
}

/// The job's settings that are not the request.
struct Job {
    kind: Kind,
    document: u64,
    mode: Value,
    model: String,
    name: String,
}

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

/// Whether an AI selection is running (for tests and the status line).
pub fn busy(app: &PhotocraftApp) -> bool {
    app.ai_select.running.is_some()
}

/// Takes over `select.subject` and `select.object` when an AI model is ready; `None` lets the
/// built-in command run. Automation keeps the built-in, deterministic commands.
pub fn intercept(app: &mut PhotocraftApp, command: &str, params: &Value) -> Option<Result<Value, String>> {
    if command == "edit.generativeFill" {
        return intercept_fill(app);
    }
    let kind = match command {
        "select.subject" => Kind::Subject,
        "select.object" => Kind::Object,
        _ => return None,
    };
    if app.automation_input || params.get("builtIn").and_then(Value::as_bool) == Some(true) {
        return None;
    }
    // A malformed Object Selection is the built-in command's to refuse, with its own message.
    let target = match kind {
        Kind::Subject | Kind::Fill => None,
        Kind::Object => Some(object_target(params)?),
    };
    let service = app.services.ai_select.as_ref()?;
    let AiSelectStatus::Ready { model } = service.status() else { return None };
    if app.ai_select.running.is_some() {
        return Some(Ok(json!({"pending": true})));
    }
    let all_layers = params.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(false);
    let source = match app.run_command("select.aiSource", json!({"sampleAllLayers": all_layers, "hint": kind == Kind::Subject})) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    let png = match source.get("png").and_then(Value::as_str).map(|s| base64::engine::general_purpose::STANDARD.decode(s)) {
        Some(Ok(bytes)) => bytes,
        _ => return Some(Err(tl!("The picture could not be prepared for the AI model").into())),
    };
    let numbers =
        |key: &str| -> Vec<f64> { source.get(key).and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default() };
    let (area, click) = match target {
        None => {
            let b = numbers("box");
            let area = (b.len() == 4 && b[0] < b[2] && b[1] < b[3]).then(|| [b[0] as f32, b[1] as f32, b[2] as f32, b[3] as f32]);
            (area, None)
        }
        Some(target) => {
            let [cx, cy, cw, ch] = match numbers("canvas")[..] {
                [x, y, w, h] if w > 0.0 && h > 0.0 => [x, y, w, h],
                _ => return Some(Err(tl!("The picture could not be prepared for the AI model").into())),
            };
            let fx = |x: f64| ((x - cx) / cw).clamp(0.0, 1.0) as f32;
            let fy = |y: f64| ((y - cy) / ch).clamp(0.0, 1.0) as f32;
            match target {
                ObjectTarget::Rect([x, y, w, h]) => {
                    let area = [fx(x), fy(y), fx(x + w), fy(y + h)];
                    if area[0] >= area[2] || area[1] >= area[3] {
                        // Entirely off the canvas: the built-in command says so.
                        return None;
                    }
                    (Some(area), None)
                }
                ObjectTarget::Point([x, y]) => (None, Some([fx(x), fy(y)])),
            }
        }
    };
    let document = source.get("document").and_then(Value::as_u64).unwrap_or(0);
    let mode = params.get("mode").cloned().unwrap_or_else(|| json!("replace"));
    let request = AiSelectRequest { png, area, click, prompt: PROMPT.into() };
    Some(begin(app, Request::Select(request), Job { kind, document, mode, model, name: String::new() }, None))
}

/// Edit › Generative Fill: with an image model ready, the prompt; without one, Preferences on
/// the page that sets one up. Headless, or with no selection, the engine's command answers.
fn intercept_fill(app: &mut PhotocraftApp) -> Option<Result<Value, String>> {
    if app.automation_input {
        return None;
    }
    let service = app.services.ai_select.as_ref()?;
    app.session.active()?.doc.selection.as_ref()?;
    if app.ai_select.running.is_some() {
        return Some(Ok(json!({"pending": true})));
    }
    match service.fill_status() {
        AiSelectStatus::Ready { model } => {
            let message = crate::i18n::fmt(tl!("Describe what should appear in the selection. {model} paints it as a new layer."), &[("model", &model)]);
            let fields = json!({MARK: true, STAGE: PROMPT_STAGE, "__label": "Generative Fill", "message": message, "prompt": "", "model": model});
            let id = app.ui.open_dialog(DialogKind::Command, fields.as_object().cloned().unwrap_or_default());
            Some(Ok(json!({"dialog": id})))
        }
        AiSelectStatus::Unavailable { reason } => {
            let id = crate::prefs_ui::open_preferences(app, "integrations");
            app.ui.status = crate::i18n::fmt(tl!("{reason} Choose one under AI models."), &[("reason", &reason)]);
            app.ui.status_error = false;
            Some(Ok(json!({"dialog": id, "needsModel": true})))
        }
    }
}

const STAGE: &str = "stage";
const PROMPT_STAGE: &str = "prompt";
/// What an image model is asked when the person leaves the description empty.
const BLEND_PROMPT: &str = "Fill the transparent area so it blends naturally with the rest of the picture.";

fn is_prompt(fields: &Map<String, Value>) -> bool {
    fields.get(STAGE).and_then(Value::as_str) == Some(PROMPT_STAGE)
}

/// The dialog's OK button: Generate on the prompt, Send on the consent question.
pub fn ok_label(fields: &Map<String, Value>) -> &'static str {
    if is_prompt(fields) { tl!("Generate") } else { tl!("Send") }
}

/// Generative Fill's prompt was confirmed: the picture and the selection go to the model.
fn confirm_fill(app: &mut PhotocraftApp, fields: &Map<String, Value>) -> Result<Value, String> {
    let model = fields.get("model").and_then(Value::as_str).unwrap_or_default().to_string();
    let typed = fields.get("prompt").and_then(Value::as_str).unwrap_or_default().trim().to_string();
    let source = app.run_command("edit.aiFillSource", json!({}))?;
    let decode = |key: &str| source.get(key).and_then(Value::as_str).and_then(|s| base64::engine::general_purpose::STANDARD.decode(s).ok());
    let (Some(png), Some(mask)) = (decode("png"), decode("mask")) else {
        return Err(tl!("The picture could not be prepared for the AI model").into());
    };
    let document = source.get("document").and_then(Value::as_u64).unwrap_or(0);
    let name = if typed.is_empty() { "Generative Fill".to_string() } else { typed.chars().take(60).collect() };
    let prompt = if typed.is_empty() { BLEND_PROMPT.to_string() } else { typed };
    let request = AiFillRequest { png, mask, prompt };
    begin(app, Request::Fill(request), Job { kind: Kind::Fill, document, mode: Value::Null, model, name }, None)
}

/// Where Object Selection asked: its rectangle `[x, y, w, h]` or its point `[x, y]`, in canvas
/// pixels, as `select.object` takes them.
#[derive(Clone, Copy, Debug)]
enum ObjectTarget {
    Rect([f64; 4]),
    Point([f64; 2]),
}

fn object_target(params: &Value) -> Option<ObjectTarget> {
    let numbers = |key: &str| -> Option<Vec<f64>> {
        let v: Vec<f64> = params.get(key)?.as_array()?.iter().filter_map(Value::as_f64).collect();
        v.iter().all(|n| n.is_finite()).then_some(v)
    };
    if let Some(r) = numbers("rect") {
        return match r[..] {
            [x, y, w, h] if w > 0.0 && h > 0.0 => Some(ObjectTarget::Rect([x, y, w, h])),
            _ => None,
        };
    }
    match numbers("point")?[..] {
        [x, y] => Some(ObjectTarget::Point([x, y])),
        _ => None,
    }
}

fn begin(app: &mut PhotocraftApp, request: Request, job: Job, consent: Option<AiConsentTerm>) -> Result<Value, String> {
    let Job { kind, document, mode, model, name } = job;
    let Some(service) = app.services.ai_select.as_ref() else { return Err(tl!("AI selection is not available").into()) };
    let started = match &request {
        Request::Select(r) => service.start(r.clone(), consent),
        Request::Fill(r) => service.start_fill(r.clone(), consent),
    };
    match started {
        Ok(AiSelectStart::Started(job)) => {
            app.ai_select.running = Some(Running { job, kind, document, mode, model: model.clone(), name });
            app.ui.status = crate::i18n::fmt(kind.working(), &[("model", &model)]);
            app.ui.status_error = false;
            Ok(json!({"pending": true, "model": model}))
        }
        Ok(AiSelectStart::NeedsConsent(ask)) => {
            if consent.is_some() {
                return Err(tl!("The AI model still needs your agreement. Please try again.").into());
            }
            app.ai_select.awaiting = Some(Awaiting { request, kind, document, mode, model: model.clone(), name });
            if let Some(open) = app.ui.dialogs.iter().find(|d| owns(&d.fields)) {
                return Ok(json!({"dialog": open.id, "consentPending": true}));
            }
            let message = crate::i18n::fmt(kind.sends(), &[("model", &model), ("destination", &ask.destination)]);
            let fields = json!({MARK: true, "__label": kind.label(), "message": message, "items": ask.items, "term": AiConsentTerm::ThisTime.id()});
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
    if is_prompt(fields) {
        ui.add_space(6.0);
        let mut prompt = fields.get("prompt").and_then(Value::as_str).unwrap_or_default().to_string();
        let hint = tl!("For example: a red umbrella");
        crate::widgets::text_field(ui, ui.scope_id().with("ai-fill-prompt"), &mut prompt, hint, 360.0, false);
        ui.add_space(4.0);
        ui.label(egui::RichText::new(tl!("Leave it empty to fill the selection to match its surroundings.")).small().weak());
        fields.insert("prompt".into(), json!(prompt));
        return;
    }
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
    if is_prompt(fields) {
        return confirm_fill(app, fields);
    }
    let term = fields.get("term").and_then(Value::as_str).and_then(AiConsentTerm::parse).unwrap_or(AiConsentTerm::ThisTime);
    let Awaiting { request, kind, document, mode, model, name } = app.ai_select.awaiting.take().ok_or(tl!("Nothing is waiting to be sent"))?;
    if app.session.active().map(|d| d.doc.id.0) != Some(document) {
        return Err(kind.again().into());
    }
    begin(app, request, Job { kind, document, mode, model, name }, Some(term))
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
            let kind = running.kind;
            app.ai_select.running = None;
            app.ui.status = kind.cancelled().into();
            app.ui.status_error = false;
        }
        AiJobPoll::Failed(message) => {
            app.ai_select.running = None;
            app.ui.status = message;
            app.ui.status_error = true;
        }
        AiJobPoll::Done { png, matched, name } => {
            let Some(Running { kind, document, mode, model, name: layer_name, .. }) = app.ai_select.running.take() else { return };
            if !matched {
                app.ui.status = kind.nothing().into();
                app.ui.status_error = false;
                return;
            }
            if kind == Kind::Fill {
                let params = json!({"png": base64::engine::general_purpose::STANDARD.encode(png), "document": document, "name": layer_name});
                match app.run_command("layer.fromAiImage", params) {
                    Ok(_) => {
                        app.ui.status = crate::i18n::fmt(kind.done(), &[("model", &model)]);
                        app.ui.status_error = false;
                    }
                    Err(e) => {
                        app.ui.status = e;
                        app.ui.status_error = true;
                    }
                }
                ctx.request_repaint();
                return;
            }
            let params = json!({
                "png": base64::engine::general_purpose::STANDARD.encode(png),
                "document": document,
                "mode": mode,
                "label": kind.label(),
            });
            match app.run_command("select.fromMask", params) {
                Ok(v) if v.get("changed").and_then(Value::as_bool) == Some(false) => {
                    app.ui.status = kind.nothing().into();
                    app.ui.status_error = false;
                }
                Ok(_) => {
                    app.ui.status = match name {
                        Some(name) => crate::i18n::fmt(tl!("Selected {name} with {model}"), &[("name", &name), ("model", &model)]),
                        None => crate::i18n::fmt(kind.done(), &[("model", &model)]),
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
