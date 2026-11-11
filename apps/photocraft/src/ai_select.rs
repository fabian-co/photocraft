//! Select Subject, Object Selection and Generative Fill on the AI models the person chose for the
//! Craft apps, through the shared Craft AI layer (`storytold/craft-ai-integrations`).
//!
//! The choice of model, the connections and the keys are shared with the other Craft apps (the
//! launcher's Set up AI window writes them). This app ships the built-in engines that run a
//! selection model on this computer (SAM 2.1 on rten) plus the hosted adapters of
//! `ProviderFactory::standard`. Consent for anything that leaves the computer is kept per app in
//! the layer's consent file. Nothing here reads, stores or prints a key.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::Duration;

use craft_ai_client::{AiClient, Plan, RoleStatus};
use craft_ai_core::segment::{self, Area, Click};
use craft_ai_core::{AppId, ConsentLedger, ConsentTerm, ErrorKind, Event, InputRole, JobState, Payload, PayloadKind, Request, Role};
use photocraft_ui_egui::ai_select::{
    AiConsentAsk, AiConsentTerm, AiFillRequest, AiJobPoll, AiModelChoice, AiSelect, AiSelectJob, AiSelectRequest, AiSelectStart, AiSelectStatus,
};

mod settings;

/// This app's id in the shared store: it scopes consent and a per-app choice of model.
const APP_ID: &str = "photocraft";
/// The suite's Select Subject tool in the shared catalogue.
const FEATURE: &str = "select-subject";
/// The suite's Generative Fill tool in the shared catalogue.
const FILL_FEATURE: &str = "generative-fill";
/// How often the worker asks the model how it is doing.
const POLL_EVERY: Duration = Duration::from_millis(100);

/// The service, or `None` when the shared AI folder cannot be read (the app works without it).
pub fn service() -> Option<Box<dyn AiSelect>> {
    let role_of = |id: &str| craft_ai_features::builtin().into_iter().find(|f| f.id == id).and_then(|f| f.role());
    let role = role_of(FEATURE)?;
    let fill_role = role_of(FILL_FEATURE);
    let app = AppId::new(APP_ID).ok()?;
    match AiClient::standard(app.clone()) {
        Ok(client) => Some(Box::new(CraftAiSelect { client: Arc::new(client), app, role, fill_role, settings: settings::Settings::default() })),
        Err(error) => {
            log::warn!("AI Select Subject is off: {error}");
            None
        }
    }
}

struct CraftAiSelect {
    client: Arc<AiClient>,
    app: AppId,
    role: Role,
    /// Generative Fill, when the catalogue has it.
    fill_role: Option<Role>,
    /// Edit › Preferences › Integrations › AI models.
    settings: settings::Settings,
}

impl AiSelect for CraftAiSelect {
    fn status(&self) -> AiSelectStatus {
        match self.client.status(&self.role) {
            RoleStatus::Ready(ready) => AiSelectStatus::Ready { model: ready.model_label },
            RoleStatus::Unavailable { reason } => AiSelectStatus::Unavailable { reason },
        }
    }

    fn start(&self, request: AiSelectRequest, consent: Option<AiConsentTerm>) -> Result<AiSelectStart, String> {
        let plan = self.client.plan(&self.role, self.request(request)?).map_err(|e| e.message)?;
        self.launch(plan, consent, Wants::Mask)
    }

    fn fill_status(&self) -> AiSelectStatus {
        let Some(role) = &self.fill_role else {
            return AiSelectStatus::Unavailable { reason: "This version of PhotoCraft has no Generative Fill tool.".into() };
        };
        match self.client.status(role) {
            RoleStatus::Ready(ready) => AiSelectStatus::Ready { model: ready.model_label },
            RoleStatus::Unavailable { reason } => AiSelectStatus::Unavailable { reason },
        }
    }

    fn start_fill(&self, r: AiFillRequest, consent: Option<AiConsentTerm>) -> Result<AiSelectStart, String> {
        let fill = self.fill_role.as_ref().ok_or("This version of PhotoCraft has no Generative Fill tool.")?;
        // Without a mask (nothing selected) it is an instruction for the whole picture: the same
        // chosen model, asked for `image.edit`, which takes no mask, instead of `image.inpaint`.
        let edit;
        let role = if r.mask.is_empty() {
            edit = edit_role(fill)?;
            &edit
        } else {
            fill
        };
        let picture = Payload::inline("image/png", r.png).map_err(|e| e.message)?;
        let mut request = Request::new(role.capability.clone()).with_input(InputRole::Source, picture);
        if !r.mask.is_empty() {
            request = request.with_input(InputRole::Mask, Payload::inline("image/png", r.mask).map_err(|e| e.message)?);
        }
        let request = request.with_input(InputRole::Prompt, Payload::text(r.prompt));
        // References follow the picture, as images 2, 3 and 4 the prompt names.
        let mut request = request;
        for png in r.references {
            request = request.with_input(InputRole::Reference, Payload::inline("image/png", png).map_err(|e| e.message)?);
        }
        // The task bar's choices, as OpenAI's images endpoints take them.
        let request = match r.size {
            Some([w, h]) => request.with_param("size", serde_json::json!(format!("{w}x{h}"))),
            None => request,
        };
        let request = match r.quality {
            Some(quality) => request.with_param("quality", serde_json::json!(quality)),
            None => request,
        };
        let plan = self.client.plan(role, request).map_err(|e| e.message)?;
        self.launch(plan, consent, Wants::Picture)
    }

    fn fill_models(&self) -> Vec<AiModelChoice> {
        let Some(role) = &self.fill_role else { return Vec::new() };
        let current = match self.client.status(role) {
            RoleStatus::Ready(ready) => Some((ready.connection, ready.model)),
            RoleStatus::Unavailable { .. } => None,
        };
        let Ok(choices) = self.client.choices(role) else { return Vec::new() };
        let several = choices.iter().any(|c| choices.iter().any(|o| o.connection != c.connection));
        choices
            .into_iter()
            .map(|c| AiModelChoice {
                selected: current.as_ref().is_some_and(|(conn, model)| *conn == c.connection && *model == c.model),
                label: if several { format!("{} · {}", c.model_label, c.connection_label) } else { c.model_label },
                connection: c.connection.as_str().to_owned(),
                model: c.model.as_str().to_owned(),
            })
            .collect()
    }

    fn choose_fill_model(&self, connection: &str, model: &str) -> Result<(), String> {
        let role = self.fill_role.as_ref().ok_or("This version of PhotoCraft has no Generative Fill tool.")?;
        let connection = craft_ai_core::ConnectionId::new(connection).map_err(|e| e.to_string())?;
        let model = craft_ai_core::ModelId::new(model).map_err(|e| e.to_string())?;
        settings::choose(&self.app, role, &connection, &model)
    }

    fn preferences_ui(&mut self, ui: &mut egui::Ui) {
        let mut tools = vec![(self.role.clone(), "Select Subject")];
        if let Some(fill) = &self.fill_role {
            tools.push((fill.clone(), "Generative Fill"));
        }
        self.settings.ui(ui, &self.app, &self.client, &tools);
    }
}

/// What the job hands back: a selection mask, or a picture.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Wants {
    Mask,
    Picture,
}

impl CraftAiSelect {
    /// Asks first when the plan sends anything off this computer, then starts the job on a worker.
    fn launch(&self, plan: Plan, consent: Option<AiConsentTerm>, wants: Wants) -> Result<AiSelectStart, String> {
        let file = self.client.consent_file();
        let mut saved = file.as_ref().map(|f| f.load_or_empty()).unwrap_or_default();
        let ledger = if plan.needs_consent(&saved) {
            let Some(term) = consent else { return Ok(AiSelectStart::NeedsConsent(ask(&plan))) };
            let term = consent_term(term);
            let ledger = saved.agree(plan.disclosure(), term, now());
            // "This time" leaves nothing on disk.
            if term.remembers()
                && let Some(file) = &file
                && let Err(error) = file.save(&saved)
            {
                log::warn!("the AI consent could not be saved: {error}");
            }
            ledger
        } else {
            saved
        };
        Ok(AiSelectStart::Started(Box::new(Worker::spawn(self.client.clone(), plan, ledger, wants))))
    }

    /// The `image.segment` request: the picture and the box, or the click (the middle of the
    /// picture when there is neither), as the shared contract spells them. No words: every built-in route reads
    /// a box or clicks, and the ones that select only by clicking (SAM 2.1) refuse a description,
    /// so `r.prompt` is not sent.
    fn request(&self, r: AiSelectRequest) -> Result<Request, String> {
        let picture = Payload::inline("image/png", r.png).map_err(|e| e.message)?;
        let request = Request::new(self.role.capability.clone()).with_input(InputRole::Source, picture);
        Ok(match r.area {
            Some([x0, y0, x1, y1]) => segment::with_area(request, Area { x0, y0, x1, y1 }),
            None => {
                let [x, y] = r.click.unwrap_or([0.5, 0.5]);
                segment::with_clicks(request, &[Click { x, y, positive: true }])
            }
        })
    }
}

/// Generative Fill's role asking its model for `image.edit` (an instruction for the whole
/// picture, no mask) instead of `image.inpaint`: the same id, so the same chosen model.
fn edit_role(fill: &Role) -> Result<Role, String> {
    let capability = craft_ai_core::CapabilityId::parse("image.edit").map_err(|e| e.to_string())?;
    Ok(Role { capability, inputs: vec![InputRole::Source, InputRole::Prompt], ..fill.clone() })
}

fn consent_term(term: AiConsentTerm) -> ConsentTerm {
    match term {
        AiConsentTerm::ThisTime => ConsentTerm::ThisTime,
        AiConsentTerm::OneHour => ConsentTerm::OneHour,
        AiConsentTerm::OneDay => ConsentTerm::OneDay,
        AiConsentTerm::UntilTurnedOff => ConsentTerm::UntilRevoked,
    }
}

/// The plan's disclosure in words: where it goes and what is sent.
fn ask(plan: &Plan) -> AiConsentAsk {
    let d = plan.disclosure();
    let items = d
        .items
        .iter()
        .map(|item| {
            let kind = match item.kind {
                PayloadKind::Text => "Text",
                PayloadKind::Image => "Picture",
                PayloadKind::Video => "Video",
                PayloadKind::Audio => "Audio",
                PayloadKind::Document => "Document",
                PayloadKind::Other => "Data",
            };
            match item.bytes {
                Some(bytes) => format!("{kind}, {}", size(bytes)),
                None => kind.to_string(),
            }
        })
        .collect();
    AiConsentAsk { destination: d.destination.clone(), items }
}

fn size(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 20 => format!("{:.1} MB", b as f64 / f64::from(1u32 << 20)),
        b if b >= 1 << 10 => format!("{} KB", b >> 10),
        b => format!("{b} bytes"),
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Starts and polls the job on its own thread (adapters are synchronous and `start` may call the
/// service), and reports to the interface through a channel.
struct Worker {
    updates: Receiver<AiJobPoll>,
    cancel: Arc<AtomicBool>,
    last: AiJobPoll,
    done: bool,
}

impl Worker {
    fn spawn(client: Arc<AiClient>, plan: Plan, ledger: ConsentLedger, wants: Wants) -> Self {
        let (tx, updates) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        std::thread::Builder::new()
            .name("ai-select-subject".into())
            .spawn(move || {
                let end = run(&client, &plan, &ledger, &stop, wants, |poll| {
                    let _ = tx.send(poll);
                });
                let _ = tx.send(end);
            })
            .map_or_else(
                |error| Self { updates: mpsc::channel().1, cancel: Arc::default(), last: AiJobPoll::Failed(format!("Could not start: {error}")), done: true },
                |_| Self { updates, cancel, last: AiJobPoll::Running(None), done: false },
            )
    }
}

fn run(client: &AiClient, plan: &Plan, ledger: &ConsentLedger, stop: &AtomicBool, wants: Wants, report: impl Fn(AiJobPoll)) -> AiJobPoll {
    let failed = |e: craft_ai_core::AiError| if e.kind == ErrorKind::Cancelled { AiJobPoll::Cancelled } else { AiJobPoll::Failed(e.message) };
    // The worker owns the plan (and the picture in it) until the job has ended.
    let job = match client.start(plan, ledger) {
        Ok(job) => job,
        Err(e) => return failed(e),
    };
    let mut outputs = Vec::new();
    let mut cancel_sent = false;
    // The tokens the model reports (`[input, output]`), for the cost shown afterwards.
    let mut usage = None;
    loop {
        if stop.load(Ordering::Relaxed) && !cancel_sent {
            cancel_sent = true;
            if job.cancel().is_err() {
                // This connection cannot stop a running job: wait for it to end, then drop the result.
                log::info!("the AI selection could not be stopped; its result will be discarded");
            }
        }
        let snapshot = match job.poll() {
            Ok(s) => s,
            Err(e) => return failed(e),
        };
        if let Some(u) = &snapshot.usage
            && let (Some(input), Some(output)) = (u.input_units, u.output_units)
        {
            usage = Some([input, output]);
        }
        outputs.extend(snapshot.events.into_iter().filter_map(|e| if let Event::Output(o) = e { Some(o) } else { None }));
        match snapshot.state {
            JobState::Queued => {}
            JobState::Running { percent } => report(AiJobPoll::Running(percent)),
            JobState::Cancelled => return AiJobPoll::Cancelled,
            JobState::Failed { error } => return failed(error),
            JobState::Succeeded if cancel_sent => return AiJobPoll::Cancelled,
            JobState::Succeeded if wants == Wants::Picture => {
                // The first picture the model returned.
                let picture = outputs.iter().find_map(|o| match &o.payload {
                    Payload::Inline { data, media_type } if media_type.kind() == PayloadKind::Image => Some(data.clone()),
                    _ => None,
                });
                return match picture {
                    Some(png) => AiJobPoll::Done { png, matched: true, name: None, usage },
                    None => AiJobPoll::Failed("The AI model returned no picture. Try another description, or another model.".into()),
                };
            }
            JobState::Succeeded => {
                let found = segment::read_found(&outputs);
                return match segment::mask(&outputs) {
                    Some(Payload::Inline { data, .. }) => {
                        AiJobPoll::Done { png: data.clone(), matched: found.as_ref().is_none_or(|f| f.matched), name: found.and_then(|f| f.name), usage }
                    }
                    _ => AiJobPoll::Failed("The AI model returned no selection. Try again, or choose another model in Set up AI.".into()),
                };
            }
        }
        std::thread::sleep(POLL_EVERY);
    }
}

impl AiSelectJob for Worker {
    fn poll(&mut self) -> AiJobPoll {
        while !self.done {
            match self.updates.try_recv() {
                Ok(update) => {
                    self.done = !matches!(update, AiJobPoll::Running(_));
                    self.last = update;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.done = true;
                    self.last = AiJobPoll::Failed("AI selection stopped unexpectedly".into());
                }
            }
        }
        self.last.clone()
    }

    fn cancel(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Generative Fill against a real model, end to end: the task bar's path through the engine, the
/// Craft AI layer and back into a layer. Costs money, so it only runs when asked, with a shared AI
/// folder to use (`CRAFT_AI_DIR`, whose Generative Fill model should be a cheap one) and:
/// `PHOTOCRAFT_LIVE_FILL_IMAGE` (a photo), `PHOTOCRAFT_LIVE_FILL_OUT` (a folder for the results),
/// `PHOTOCRAFT_LIVE_FILL_RECT` (`x,y,w,h` to select; empty: no selection), `PHOTOCRAFT_LIVE_FILL_PROMPT`
/// and optionally `PHOTOCRAFT_LIVE_FILL_REF` (a picture tagged by its file name) and
/// `PHOTOCRAFT_LIVE_FILL_NAME` (the result's file name).
#[cfg(test)]
mod live_tests {
    use base64::Engine as _;
    use serde_json::{Value, json};

    fn var(name: &str) -> Option<String> {
        std::env::var(name).ok().filter(|v| !v.is_empty())
    }

    #[test]
    #[ignore = "sends a picture to a paid AI model"]
    fn generative_fill_on_a_real_model() {
        let (Some(image), Some(out), Some(prompt)) = (var("PHOTOCRAFT_LIVE_FILL_IMAGE"), var("PHOTOCRAFT_LIVE_FILL_OUT"), var("PHOTOCRAFT_LIVE_FILL_PROMPT")) else {
            panic!("set PHOTOCRAFT_LIVE_FILL_IMAGE, PHOTOCRAFT_LIVE_FILL_OUT and PHOTOCRAFT_LIVE_FILL_PROMPT");
        };
        let name = var("PHOTOCRAFT_LIVE_FILL_NAME").unwrap_or_else(|| "result".into());
        let mut session = photocraft_engine::Session::new();
        if image == "blank" {
            session.execute("file.new", json!({"width": 1200, "height": 800})).unwrap();
        } else {
            let bytes = std::fs::read(&image).unwrap();
            let decoded = photocraft_codecs::decode(&bytes).unwrap();
            session.execute("file.new", json!({"width": decoded.width(), "height": decoded.height()})).unwrap();
            let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
            session.execute("layer.fromAiImage", json!({"png": b64, "name": "Photo"})).unwrap();
        }
        if let Some(rect) = var("PHOTOCRAFT_LIVE_FILL_RECT") {
            let n: Vec<i64> = rect.split(',').map(|v| v.trim().parse().unwrap()).collect();
            session.execute("select.rect", json!({"x": n[0], "y": n[1], "width": n[2], "height": n[3]})).unwrap();
        }
        let services = photocraft_ui_egui::Services { ai_select: super::service(), ..Default::default() };
        let mut app = photocraft_ui_egui::PhotocraftApp::new(session, services);
        let refs = var("PHOTOCRAFT_LIVE_FILL_REF")
            .map(|path| {
                let tag = std::path::Path::new(&path).file_stem().unwrap().to_string_lossy().into_owned();
                vec![(tag, std::fs::read(&path).unwrap())]
            })
            .unwrap_or_default();
        let started = photocraft_ui_egui::ai_select::generative_fill(&mut app, &prompt, Some("low"), None, refs).unwrap();
        eprintln!("started: {started}");
        assert!(started.get("consentPending").is_none(), "agree to send pictures in this AI folder first");
        let ctx = egui::Context::default();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        while photocraft_ui_egui::ai_select::busy(&app) {
            assert!(std::time::Instant::now() < deadline, "the model took too long");
            std::thread::sleep(std::time::Duration::from_millis(250));
            photocraft_ui_egui::ai_select::tick(&mut app, &ctx);
        }
        eprintln!("status: {} (error: {})", app.ui.status, app.ui.status_error);
        assert!(!app.ui.status_error, "{}", app.ui.status);
        // The new layer alone, unmasked, over its bounds: what the model drew, colour-matched.
        if let Some(d) = app.session.active()
            && let Some(layer) = d.active_layer.and_then(|id| d.doc.layer(id))
            && let Some(surface) = layer.surface()
        {
            let r = surface.content_bounds();
            let px = surface.read_region(r);
            let mut img = photocraft_codecs::Image::new(r.width(), r.height(), photocraft_codecs::ChannelLayout::Rgba, photocraft_codecs::SampleType::U8).unwrap();
            for (o, v) in img.data_mut().iter_mut().zip(px) {
                *o = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
            let png = photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &Default::default()).unwrap();
            std::fs::write(std::path::Path::new(&out).join(format!("{name}-layer.png")), png).unwrap();
            eprintln!("layer bounds {r:?}");
        }
        // The composite, at most 1600 px on its longest side.
        // Nothing to drop when nothing was selected.
        let _ = app.session.execute("select.deselect", json!({}));
        let v: Value = app.session.execute("edit.aiFillSource", json!({"crop": false, "maxSide": 1600})).unwrap();
        let png = base64::engine::general_purpose::STANDARD.decode(v["png"].as_str().unwrap()).unwrap();
        let path = std::path::Path::new(&out).join(format!("{name}.png"));
        std::fs::write(&path, png).unwrap();
        eprintln!("wrote {}", path.display());
    }
}
