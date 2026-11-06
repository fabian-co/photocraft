//! The AI block of Edit › Preferences › Integrations: which model each AI tool uses, getting the
//! files of a model that runs on this computer, and adding an online service with its key.
//!
//! It draws its own views over the shared data (`AiClient::picker`, `SetupSession`) so every
//! sentence goes through PhotoCraft's translations; the online-service form is the suite's own
//! panel (`craft_ai_egui::ConnectionPanel`). Everything is written to the shared store, so a
//! model chosen or a service added here is the same in every Craft app.
//!
//! Disk and network work (checking or downloading model files) runs on a worker thread; the
//! interface only draws and passes on what the person pressed.

use std::path::Path;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use craft_ai_client::{
    AiClient, CheckedNative, ConnectionManager, FolderDesk, GroupState, ModelDownloadConfig, ModelDownloader, NativeSetupPlan, PickerGroup, PickerModel,
    PickerView, RoleStatus, SetupBlock, SetupLink, SetupPhase, SetupProgress, SetupRequest, SetupSession, SetupTask, SetupTicket, ToolDesk,
};
use craft_ai_core::{AiError, AppId, ConnectionId, ErrorKind, Locality, ModelId, Role};
use craft_ai_egui::{ConnectionFormState, ConnectionPanel, PanelEvent};
use craft_ai_features::Feature;
use craft_ai_native::shared::SharedStore;
use egui::{RichText, Ui};
use photocraft_ui_egui::i18n::t;

/// How often the lists are read again while Preferences is open, so a choice made in another
/// Craft app shows up.
const REFRESH_EVERY: Duration = Duration::from_secs(3);
/// Preferences not drawn for this long was closed: the last answer is forgotten.
const CLOSED_AFTER: Duration = Duration::from_secs(1);

/// What the setup worker reports.
enum Report {
    Progress(SetupTicket, usize, SetupProgress),
    Done(SetupTicket, Box<Result<CheckedNative, AiError>>),
}

/// The files of one built-in model being checked or downloaded.
struct Setup {
    feature: Feature,
    provider: String,
    session: SetupSession,
    reports: Option<Receiver<Report>>,
}

/// What the person pressed this frame; applied after drawing.
enum Action {
    Choose(ConnectionId, ModelId),
    SetUp(SetupLink),
}

#[derive(Default)]
pub(crate) struct Settings {
    /// Which tool's models are shown (an index into the tools given to [`Settings::ui`]).
    tool: usize,
    view: Option<Result<PickerView, String>>,
    read_at: Option<Instant>,
    drawn_at: Option<Instant>,
    setup: Option<Setup>,
    panel: Option<ConnectionPanel>,
    /// The answer to the last thing the person did, and whether it is a problem.
    note: Option<(String, bool)>,
    /// Saved online services whose models are not known yet (saved without a test): the picker
    /// leaves them out, so they are listed here with a button to ask the service.
    unlisted: Vec<(ConnectionId, String)>,
    /// A service being asked for its models.
    asking: Option<Receiver<Result<usize, String>>>,
}

impl Settings {
    /// `tools` are the AI tools this app offers, each with what it is called here.
    pub(crate) fn ui(&mut self, ui: &mut Ui, app: &AppId, client: &AiClient, tools: &[(Role, &str)]) {
        if self.drawn_at.is_none_or(|t| t.elapsed() >= CLOSED_AFTER) {
            // Preferences was opened again: start from a fresh list and no stale answer.
            self.note = None;
            self.read_at = None;
        }
        self.drawn_at = Some(Instant::now());
        // Keep drawing while Preferences is open, so a pause in input is not taken for closing.
        ui.ctx().request_repaint_after(CLOSED_AFTER / 2);
        self.poll_asking();
        if self.panel.is_some() {
            self.panel_ui(ui);
        } else if self.setup.is_some() {
            if let Some((role, _)) = tools.get(self.tool) {
                self.setup_ui(ui, app, client, role);
            }
        } else {
            self.tools_ui(ui, app, client, tools);
        }
        if let Some((note, problem)) = &self.note {
            ui.add_space(6.0);
            if *problem {
                ui.colored_label(ui.visuals().warn_fg_color, note);
            } else {
                ui.label(note);
            }
        }
    }

    fn tools_ui(&mut self, ui: &mut Ui, app: &AppId, client: &AiClient, tools: &[(Role, &str)]) {
        if tools.len() > 1 {
            ui.horizontal_wrapped(|ui| {
                for (i, (_, name)) in tools.iter().enumerate() {
                    if ui.selectable_label(self.tool == i, t(name)).clicked() && self.tool != i {
                        self.tool = i;
                        self.read_at = None;
                        self.note = None;
                    }
                }
            });
            ui.add_space(4.0);
        }
        let Some((role, name)) = tools.get(self.tool) else { return };
        if self.read_at.is_none_or(|t| t.elapsed() >= REFRESH_EVERY) {
            self.refresh(client, role);
        }
        ui.ctx().request_repaint_after(REFRESH_EVERY);
        let name = t(name);
        match client.status(role) {
            RoleStatus::Ready(ready) => {
                ui.label(fill(t("{tool} uses {model}."), &[("tool", name), ("model", &ready.model_label)]));
            }
            RoleStatus::Unavailable { .. } if role.id.as_str() == super::FEATURE => {
                ui.label(fill(t("{tool} uses PhotoCraft's own selection: no AI model is chosen yet."), &[("tool", name)]));
            }
            RoleStatus::Unavailable { .. } => {
                ui.label(fill(t("{tool} needs an AI model: choose or set one up below."), &[("tool", name)]));
            }
        }
        ui.add_space(6.0);
        let action = match &self.view {
            Some(Ok(view)) => picker(ui, view),
            Some(Err(message)) => {
                ui.colored_label(ui.visuals().warn_fg_color, message);
                None
            }
            None => None,
        };
        let mut ask = None;
        for (id, label) in &self.unlisted {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(fill(t("{service} is saved, but its models are not known yet."), &[("service", label)]));
                let busy = self.asking.is_some();
                if ui.add_enabled(!busy, egui::Button::new(t("Find its models"))).clicked() {
                    ask = Some((id.clone(), true));
                }
                if ui.button(t("Change the key")).clicked() {
                    ask = Some((id.clone(), false));
                }
            });
        }
        match ask {
            Some((id, true)) => self.ask_models(id),
            Some((id, false)) => self.open_fix(&id),
            None => {}
        }
        ui.add_space(8.0);
        if ui.button(t("Add an online service (OpenAI, Gemini…)")).clicked() {
            self.open_panel(ConnectionFormState::new());
        }
        match action {
            Some(Action::Choose(connection, model)) => {
                self.note = Some(match desk(app).and_then(|d| d.choose(role, &connection, &model)) {
                    Ok(()) => (t("Saved. Every Craft app uses this model for this tool now.").to_owned(), false),
                    Err(e) => (e.message, true),
                });
                self.refresh(client, role);
            }
            Some(Action::SetUp(SetupLink::New { option })) => self.open_setup(&option, client),
            Some(Action::SetUp(SetupLink::Fix { connection })) => self.open_fix(&connection),
            None => {}
        }
    }

    fn refresh(&mut self, client: &AiClient, role: &Role) {
        let options = craft_ai_features::setup_options(&craft_ai_features::builtin());
        self.view = Some(client.picker(role, &options).map_err(|e| e.message));
        self.unlisted = shared()
            .and_then(|s| s.load().map_err(|e| e.to_string()))
            .map(|store| {
                store.connections.into_iter().filter(|c| c.locality == Locality::Remote && c.enabled && c.models.is_empty()).map(|c| (c.id, c.label)).collect()
            })
            .unwrap_or_default();
        self.read_at = Some(Instant::now());
    }

    /// Asks a saved service which models it has, in the background, and keeps the answer in the
    /// shared settings, so its models show up in the list.
    fn ask_models(&mut self, id: ConnectionId) {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let answer = ConnectionManager::standard().and_then(|m| m.test_saved(&id, true)).map(|r| r.models.len()).map_err(|e| e.message);
            let _ = tx.send(answer);
        });
        self.asking = Some(rx);
        self.note = Some((t("Asking the service which models it has…").to_owned(), false));
    }

    fn poll_asking(&mut self) {
        let Some(rx) = &self.asking else { return };
        let answer = match rx.try_recv() {
            Ok(answer) => answer,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => Err(t("The service could not be asked.").to_owned()),
        };
        self.asking = None;
        self.read_at = None;
        self.note = Some(match answer {
            Ok(0) => (t("The service answered, but it has no models PhotoCraft knows.").to_owned(), true),
            Ok(_) => (t("Service saved. Its models are in the list now: click one to use it.").to_owned(), false),
            Err(problem) => (fill(t("The service did not answer: {problem}. Check the key and try again."), &[("problem", &problem)]), true),
        });
    }

    /// `native:<tool>:<engine>` downloads here, `service:<preset>` opens the service form; a
    /// ComfyUI model (`local:…`) cannot be set up from the Craft apps yet.
    fn open_setup(&mut self, option: &str, client: &AiClient) {
        let mut parts = option.split(':');
        match (parts.next(), parts.next(), parts.next()) {
            (Some("native"), Some(tool), Some(provider)) => {
                let Some(feature) = craft_ai_features::builtin().into_iter().find(|f| f.id == tool) else {
                    self.note = Some((t("This model is not in this version of PhotoCraft.").to_owned(), true));
                    return;
                };
                match plan(&feature, provider, client) {
                    Ok(plan) => {
                        self.note = None;
                        self.setup = Some(Setup { feature, provider: provider.to_owned(), session: SetupSession::new(plan), reports: None });
                    }
                    Err(message) => self.note = Some((message, true)),
                }
            }
            (Some("service"), Some(_), None) => self.open_panel(ConnectionFormState::new()),
            _ => self.note = Some((t(COMFY_NOTE).to_owned(), false)),
        }
    }

    /// A saved service missing its key or switched off: its form, to fix it.
    fn open_fix(&mut self, connection: &ConnectionId) {
        let found = shared().and_then(|s| s.load().map_err(|e| e.to_string())).map(|store| store.connections.into_iter().find(|c| &c.id == connection));
        match found {
            Ok(Some(c)) => self.open_panel(ConnectionFormState::edit(&c)),
            Ok(None) => self.note = Some((t("That service is no longer saved.").to_owned(), true)),
            Err(message) => self.note = Some((message, true)),
        }
    }

    fn open_panel(&mut self, form: ConnectionFormState) {
        match ConnectionManager::standard() {
            Ok(manager) => {
                self.note = None;
                self.panel = Some(ConnectionPanel::with_form(manager, form));
            }
            Err(e) => self.note = Some((e.message, true)),
        }
    }

    fn panel_ui(&mut self, ui: &mut Ui) {
        let Some(panel) = self.panel.as_mut() else { return };
        ui.label(RichText::new(t("Online service")).strong());
        ui.label(
            RichText::new(t("Your key is kept in this computer's keychain and is only read when a job is sent. The service bills you for what you use."))
                .small()
                .weak(),
        );
        ui.add_space(4.0);
        match panel.show(ui) {
            Some(PanelEvent::Saved(id)) => {
                self.panel = None;
                self.read_at = None;
                // Save keeps only the models a Test found; ask the service now if there are none.
                let known = shared()
                    .and_then(|s| s.load().map_err(|e| e.to_string()))
                    .is_ok_and(|store| store.connections.iter().any(|c| c.id == id && !c.models.is_empty()));
                if known {
                    self.note = Some((t("Service saved. Its models are in the list now: click one to use it.").to_owned(), false));
                } else {
                    self.ask_models(id);
                }
            }
            Some(PanelEvent::Closed) => {
                self.panel = None;
                self.read_at = None;
            }
            None => {}
        }
    }

    fn setup_ui(&mut self, ui: &mut Ui, app: &AppId, client: &AiClient, role: &Role) {
        let Some(setup) = self.setup.as_mut() else { return };
        setup.drain();
        if setup.session.busy() {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
        let Some(action) = setup_view(ui, &mut setup.session) else { return };
        match action {
            SetupAction::Work(request) => setup.reports = Some(run(request)),
            SetupAction::Use(checked) => {
                self.note = Some(match desk(app).and_then(|d| d.adopt(&checked)) {
                    Ok(adopted) if adopted.bound => (t("The model is ready, and every Craft app uses it for this tool now.").to_owned(), false),
                    Ok(_) => (t("The model is ready. Your earlier choice was kept: pick the new model in the list to use it.").to_owned(), false),
                    Err(e) => (e.message, true),
                });
                self.setup = None;
                self.refresh(client, role);
            }
            SetupAction::UseSuggestedFolder => {
                let result = folders().and_then(|f| f.ensure_default().map(|_| ()).map_err(|e| e.message));
                self.review(result, client);
            }
            SetupAction::FindFiles => {
                if let Some(path) = pick_folder(t("Find model files")) {
                    let result = folders().and_then(|f| f.add(&path, None).map(|_| ()).map_err(|e| e.message));
                    self.review(result, client);
                }
            }
            SetupAction::ChangeFolder => {
                if let Some(path) = pick_folder(t("Save model files to")) {
                    let result = folders().and_then(|f| new_default(&f, &path));
                    self.review(result, client);
                }
            }
            SetupAction::Back => {
                if let Some(mut setup) = self.setup.take() {
                    setup.session.cancel();
                }
                self.refresh(client, role);
            }
        }
    }

    /// After a folder change: show a fresh plan, or what went wrong.
    fn review(&mut self, changed: Result<(), String>, client: &AiClient) {
        let Some(setup) = self.setup.as_mut() else { return };
        match changed.and_then(|()| plan(&setup.feature, &setup.provider, client)) {
            Ok(plan) => {
                // A different target cancels the old worker (`review` raises its flag).
                let _ = setup.session.review(plan);
                self.note = None;
            }
            Err(message) => self.note = Some((message, true)),
        }
    }
}

const COMFY_NOTE: &str = "This option runs through ComfyUI, which the Craft apps cannot set up yet. Choose a built-in option instead.";

// ------------------------------------------------------------------ the model list

/// Whether PhotoCraft can use or set up `group` now; the rest are listed under Other options.
fn usable(group: &PickerGroup) -> bool {
    match (&group.setup, group.state) {
        (_, GroupState::Ready) => true,
        (_, GroupState::NeedsAdapter) => false,
        (Some(SetupLink::New { option }), _) => !option.starts_with("local:"),
        (Some(SetupLink::Fix { .. }), _) => true,
        (None, _) => false,
    }
}

/// The built-in model that runs on any computer: what a person should try first.
fn recommended(group: &PickerGroup) -> bool {
    matches!(&group.setup, Some(SetupLink::New { option }) if option.starts_with("native:") && option.ends_with(":rten"))
}

fn picker(ui: &mut Ui, view: &PickerView) -> Option<Action> {
    let mut action = None;
    if let Some(current) = &view.current
        && let Some(problem) = &current.problem
    {
        ui.colored_label(
            ui.visuals().warn_fg_color,
            fill(t("Your choice, {model}, is not working: {problem}"), &[("model", &current.label), ("problem", problem)]),
        );
        ui.add_space(4.0);
    }
    let (mut main, other): (Vec<&PickerGroup>, Vec<&PickerGroup>) = view.groups.iter().partition(|g| usable(g));
    // Ready first, then the recommended download, then the rest as the layer ordered them.
    main.sort_by_key(|g| (g.state != GroupState::Ready, !recommended(g)));
    if main.is_empty() {
        ui.label(t("Nothing PhotoCraft can run does this tool yet. Add an online service below."));
    }
    for group in main {
        if let Some(found) = group_ui(ui, group) {
            action = Some(found);
        }
        ui.add_space(6.0);
    }
    if !other.is_empty() {
        egui::CollapsingHeader::new(t("Other options (not available in PhotoCraft yet)")).id_salt(("ai-other", view.tool.as_str())).default_open(false).show(
            ui,
            |ui| {
                for group in other {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new(tr_text(&group.title)).weak());
                        if let Some(size) = group.size_bytes {
                            ui.label(RichText::new(craft_ai_egui::text::bytes(size)).small().weak());
                        }
                    });
                    let why = match (&group.setup, group.state) {
                        (_, GroupState::NeedsAdapter) => t("PhotoCraft does not include what this option needs.").to_owned(),
                        _ => t(COMFY_NOTE).to_owned(),
                    };
                    ui.label(RichText::new(why).small().weak());
                    ui.add_space(4.0);
                }
            },
        );
    }
    action
}

fn group_ui(ui: &mut Ui, group: &PickerGroup) -> Option<Action> {
    let mut action = None;
    let ready = group.state == GroupState::Ready;
    ui.horizontal_wrapped(|ui| {
        let title = RichText::new(tr_text(&group.title)).strong();
        ui.label(if ready { title } else { title.weak() });
        if recommended(group) {
            ui.label(RichText::new(t("Recommended")).small().strong().color(ui.visuals().hyperlink_color));
        }
        if let Some(locality) = group.locality {
            let place = match locality {
                Locality::Local => t("Stays on this computer"),
                Locality::Remote => t("Sent to the service"),
            };
            ui.label(RichText::new(place).small().weak());
        }
        if let Some(size) = group.size_bytes {
            ui.label(RichText::new(fill(t("Download {size}"), &[("size", &craft_ai_egui::text::bytes(size))])).small().weak());
        }
        if let Some(link) = &group.setup {
            let label = match (link, group.state) {
                (SetupLink::New { .. }, _) => t("Set up"),
                (SetupLink::Fix { .. }, GroupState::NeedsKey) => t("Add key"),
                (SetupLink::Fix { .. }, _) => t("Switch on"),
            };
            if ui.button(label).clicked() {
                action = Some(Action::SetUp(link.clone()));
            }
        }
    });
    if !group.note.is_empty() {
        ui.label(RichText::new(tr_text(&group.note)).small().weak());
    }
    ui.add_enabled_ui(ready, |ui| {
        for model in &group.models {
            if let Some(found) = model_row(ui, model) {
                action = Some(found);
            }
        }
    });
    action
}

fn model_row(ui: &mut Ui, model: &PickerModel) -> Option<Action> {
    let mut text = tr_text(&model.label);
    if model.selected {
        text = format!("✔ {text}  ({})", t("in use"));
    }
    let job = {
        let mut job = egui::text::LayoutJob::default();
        let style = ui.style();
        job.append(&text, 0.0, egui::TextFormat::simple(egui::FontSelection::Default.resolve(style), ui.visuals().strong_text_color()));
        if !model.description.is_empty() {
            job.append(
                &format!("\n{}", tr_text(&model.description)),
                0.0,
                egui::TextFormat::simple(egui::TextStyle::Small.resolve(style), ui.visuals().weak_text_color()),
            );
        }
        job
    };
    let clicked = ui.with_layout(egui::Layout::top_down_justified(egui::Align::LEFT), |ui| ui.selectable_label(model.selected, job).clicked()).inner;
    (clicked && !model.selected).then(|| Action::Choose(model.connection.clone(), model.model.clone()))
}

// ------------------------------------------------------------------ the download screen

enum SetupAction {
    Work(SetupRequest),
    Use(CheckedNative),
    FindFiles,
    ChangeFolder,
    UseSuggestedFolder,
    Back,
}

/// The built-in model's setup, in the order the shared session keeps: where the files go, the
/// licences, one start button (check what is on disk, or download), then use the checked files.
fn setup_view(ui: &mut Ui, session: &mut SetupSession) -> Option<SetupAction> {
    let plan = session.plan().clone();
    ui.label(RichText::new(fill(t("Set up {model}"), &[("model", &tr_text(&plan.route().label))])).strong());
    if let Some(note) = &plan.route().note {
        ui.label(RichText::new(tr_text(note)).small().weak());
    }
    ui.label(RichText::new(t("The model is shared with every Craft app that has this tool.")).small().weak());
    ui.add_space(6.0);
    if let Some(checked) = session.checked() {
        ui.label(t("The files are checked and ready."));
        ui.label(RichText::new(checked.weights().to_string_lossy()).small().weak());
        ui.add_space(4.0);
        let mut action = None;
        ui.horizontal(|ui| {
            if ui.button(t("Use this model")).clicked() {
                action = Some(SetupAction::Use(checked.clone()));
            }
            if ui.button(t("Back")).clicked() {
                action = Some(SetupAction::Back);
            }
        });
        return action;
    }
    ui.label(fill(t("Download {size}"), &[("size", &craft_ai_egui::text::bytes(plan.download_bytes()))]));
    match plan.destination() {
        Some(destination) => {
            ui.label(t("Model files are saved to:"));
            ui.label(RichText::new(destination.to_string_lossy()).small());
        }
        None => {
            ui.label(t("Choose where model files are saved before downloading."));
            if let Some(suggested) = craft_ai_native::paths::suggested_models_dir() {
                ui.label(RichText::new(fill(t("Suggested folder: {folder}"), &[("folder", &suggested.to_string_lossy())])).small());
                if ui.add_enabled(plan.available() && !session.busy(), egui::Button::new(t("Use the suggested folder"))).clicked() {
                    return Some(SetupAction::UseSuggestedFolder);
                }
            }
        }
    }
    let mut licences = Vec::new();
    for file in &plan.route().models {
        if !licences.contains(&file.licence) {
            licences.push(file.licence.clone());
        }
    }
    for licence in licences {
        match licence.url.as_deref().filter(|url| url.starts_with("https://")) {
            Some(url) => {
                ui.hyperlink_to(fill(t("Read the licence: {name}"), &[("name", &licence.name)]), url);
            }
            None => {
                ui.label(fill(t("Licence: {name}"), &[("name", &licence.name)]));
            }
        }
        if let Some(note) = &licence.note {
            ui.label(RichText::new(tr_text(note)).small().weak());
        }
    }
    ui.add_space(4.0);
    if session.busy() {
        if let Some((file, progress)) = session.progress() {
            let verb = match progress.phase {
                SetupPhase::Downloading => t("Downloading file {n} of {total}"),
                SetupPhase::Verifying => t("Checking file {n} of {total}"),
            };
            ui.label(fill(verb, &[("n", &(file + 1).to_string()), ("total", &plan.route().models.len().to_string())]));
            let fraction = if progress.total == 0 { 0.0 } else { (progress.done as f64 / progress.total as f64).clamp(0.0, 1.0) as f32 };
            ui.add(egui::ProgressBar::new(fraction).desired_width(260.0).show_percentage());
        } else {
            ui.label(t("Working on the model files…"));
        }
        if ui.button(t("Cancel")).clicked() {
            session.cancel();
        }
        return None;
    }
    if let Some(kind) = session.problem() {
        ui.colored_label(ui.visuals().warn_fg_color, problem_text(kind));
    }
    if session.blocker() == Some(SetupBlock::NeedsFreshFolder) {
        ui.label(t("These files did not pass their check. Choose a new, empty folder for the download; the old files are left alone."));
    }
    if !plan.available() {
        ui.label(t("PhotoCraft can't use this option yet. Choose another."));
    }
    if let Some(directory) = plan.existing_directory() {
        ui.label(t("Model files already on this computer:"));
        ui.label(RichText::new(directory.to_string_lossy()).small());
    }
    let task = session.next_task();
    if plan.needs_agreement() && task == SetupTask::Download {
        let mut accepted = session.accepted();
        ui.add_enabled_ui(plan.available(), |ui| {
            ui.checkbox(&mut accepted, t("I accept the model's licences"));
        });
        session.accept_licences(accepted);
    }
    let label = match task {
        SetupTask::CheckExisting => t("Check the files"),
        SetupTask::Download => t("Download the model"),
    };
    let mut action = None;
    ui.horizontal_wrapped(|ui| {
        if ui.add_enabled(session.blocker().is_none(), egui::Button::new(label)).clicked() {
            action = session.start().map(SetupAction::Work);
        }
        if ui.add_enabled(plan.available(), egui::Button::new(t("Find model files…"))).clicked() {
            action = Some(SetupAction::FindFiles);
        }
        if ui.add_enabled(plan.available(), egui::Button::new(t("Change folder…"))).clicked() {
            action = Some(SetupAction::ChangeFolder);
        }
        if ui.button(t("Back")).clicked() {
            action = Some(SetupAction::Back);
        }
    });
    action
}

/// Safe wording for a failed setup; a source's own error text is never shown.
fn problem_text(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::Auth => t("Access to these model files was refused."),
        ErrorKind::Cancelled => t("Setup was cancelled. A partial download carries on next time."),
        ErrorKind::Network | ErrorKind::RateLimited => t("The download could not finish. Check your connection and try again."),
        ErrorKind::InvalidRequest => t("The model files could not be checked. Look at the files and the folder."),
        ErrorKind::Unsupported => t("PhotoCraft can't use this option yet. Choose another."),
        _ => t("The model files could not be prepared. Nothing was switched on; you can try again."),
    }
}

impl Setup {
    fn drain(&mut self) {
        let Some(reports) = &self.reports else { return };
        loop {
            match reports.try_recv() {
                Ok(Report::Progress(ticket, file, progress)) => {
                    self.session.report_progress(ticket, file, progress);
                }
                Ok(Report::Done(ticket, result)) => {
                    self.session.complete(ticket, *result);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.reports = None;
                    // A worker that died without answering leaves the session busy: free it.
                    if self.session.busy() {
                        self.session.cancel();
                    }
                    break;
                }
            }
        }
    }
}

/// Checks or downloads on a worker thread. No token: the built-in models' files are public.
fn run(request: SetupRequest) -> Receiver<Report> {
    let (tx, rx) = mpsc::channel();
    let ticket = request.ticket();
    let spawned = std::thread::Builder::new().name("ai-model-setup".into()).spawn(move || {
        let downloader = ModelDownloader::new(ModelDownloadConfig::default());
        let progress_tx = tx.clone();
        let result = request.run(&downloader, None, &mut |file, progress| {
            let _ = progress_tx.send(Report::Progress(ticket, file, progress));
        });
        let _ = tx.send(Report::Done(ticket, Box::new(result)));
    });
    if let Err(error) = spawned {
        log::warn!("the AI model setup could not start: {error}");
    }
    rx
}

// ------------------------------------------------------------------ helpers

/// `{name}` placeholders filled in.
fn fill(template: &str, values: &[(&str, &str)]) -> String {
    values.iter().fold(template.to_owned(), |s, (k, v)| s.replace(&format!("{{{k}}}"), v))
}

/// A sentence from the shared layer, translated sentence by sentence where PhotoCraft has the
/// words (titles, notes, descriptions); anything else stays as the layer wrote it.
fn tr_text(text: &str) -> String {
    let whole = t(text);
    if !std::ptr::eq(whole, text) || !text.contains(". ") {
        return whole.to_owned();
    }
    text.split_inclusive(". ")
        .map(|part| {
            let trimmed = part.trim_end();
            let translated = t(trimmed);
            if std::ptr::eq(translated, trimmed) { part.to_owned() } else { format!("{translated} ") }
        })
        .collect::<String>()
        .trim_end()
        .to_owned()
}

fn shared() -> Result<SharedStore, String> {
    SharedStore::at_default_location().map_err(|e| e.to_string())
}

fn folders() -> Result<FolderDesk, String> {
    shared().map(FolderDesk::new)
}

/// Choices made here are the suite's: the same store every Craft app reads.
fn desk(app: &AppId) -> Result<ToolDesk, AiError> {
    let client = AiClient::standard(app.clone())?;
    let shared = SharedStore::at_default_location().map_err(|e| AiError::invalid(e.to_string()))?;
    Ok(ToolDesk::new(client, shared, None))
}

fn plan(feature: &Feature, provider: &str, client: &AiClient) -> Result<NativeSetupPlan, String> {
    folders()?.prepare_native_setup(feature, provider, client).map_err(|e| e.message)
}

/// A new empty folder the person chose becomes where model files are saved.
fn new_default(folders: &FolderDesk, path: &Path) -> Result<(), String> {
    let added = folders.add_new(path, None).map_err(|e| e.message)?;
    folders.set_default(&added.folder.id).map_err(|e| e.message)
}

fn pick_folder(title: &str) -> Option<std::path::PathBuf> {
    rfd::FileDialog::new().set_title(title).pick_folder()
}
