//! The AI block of Edit › Preferences › Integrations: which model Select Subject uses, and getting
//! the files of a model that runs on this computer. It is drawn with the suite's shared widgets
//! (`craft-ai-egui`), so it reads and behaves like the launcher's Set up AI window, and it writes
//! the same shared store: a model chosen here is the choice in every Craft app.
//!
//! Disk and network work (checking or downloading model files) runs on a worker thread; the
//! interface only draws the session and passes on what the person pressed.

use std::path::Path;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use craft_ai_client::{
    AiClient, CheckedNative, FolderDesk, ModelDownloadConfig, ModelDownloader, NativeSetupPlan, PickerView, RoleStatus, SetupLink, SetupProgress, SetupRequest,
    SetupSession, SetupTicket, ToolDesk,
};
use craft_ai_core::{AiError, AppId, Role};
use craft_ai_egui::{NativeSetupAction, PickerAction, model_picker, native_model_setup};
use craft_ai_features::Feature;
use craft_ai_native::shared::SharedStore;

/// How often the list is read again while Preferences is open, so a choice made in the launcher
/// or another app shows up.
const REFRESH_EVERY: Duration = Duration::from_secs(3);

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

#[derive(Default)]
pub(super) struct Settings {
    view: Option<Result<PickerView, String>>,
    read_at: Option<Instant>,
    setup: Option<Setup>,
    /// The answer to the last thing the person did, and whether it is a problem.
    note: Option<(String, bool)>,
}

impl Settings {
    pub(super) fn ui(&mut self, ui: &mut egui::Ui, app: &AppId, client: &AiClient, role: &Role) {
        if self.setup.is_some() {
            self.setup_ui(ui, app, client, role);
        } else {
            self.picker_ui(ui, app, client, role);
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

    fn picker_ui(&mut self, ui: &mut egui::Ui, app: &AppId, client: &AiClient, role: &Role) {
        if self.read_at.is_none_or(|t| t.elapsed() >= REFRESH_EVERY) {
            self.refresh(client, role);
        }
        ui.ctx().request_repaint_after(REFRESH_EVERY);
        match client.status(role) {
            RoleStatus::Ready(ready) => ui.label(format!("Select Subject uses {}.", ready.model_label)),
            RoleStatus::Unavailable { reason } => ui.label(format!("Select Subject uses the built-in selection: {reason}")),
        };
        ui.add_space(4.0);
        let action = match &self.view {
            Some(Ok(view)) => model_picker(ui, view),
            Some(Err(message)) => {
                ui.colored_label(ui.visuals().warn_fg_color, message);
                None
            }
            None => None,
        };
        match action {
            Some(PickerAction::Choose { connection, model }) => {
                self.note = Some(match desk(app).and_then(|d| d.choose(role, &connection, &model)) {
                    Ok(()) => ("Saved. Select Subject now uses this model in every Craft app.".into(), false),
                    Err(e) => (e.message, true),
                });
                self.refresh(client, role);
            }
            Some(PickerAction::SetUp(SetupLink::New { option })) => self.open_setup(&option, client),
            Some(PickerAction::SetUp(SetupLink::Fix { .. })) => {
                self.note = Some(("Fix this connection in the Craft launcher's Set up AI window, then come back here.".into(), false));
            }
            None => {}
        }
    }

    fn refresh(&mut self, client: &AiClient, role: &Role) {
        let options = craft_ai_features::setup_options(&craft_ai_features::builtin());
        self.view = Some(client.picker(role, &options).map_err(|e| e.message));
        self.read_at = Some(Instant::now());
    }

    /// `native:<tool>:<engine>` is set up here; services and ComfyUI models in the launcher.
    fn open_setup(&mut self, option: &str, client: &AiClient) {
        let mut parts = option.split(':');
        let (Some("native"), Some(tool), Some(provider)) = (parts.next(), parts.next(), parts.next()) else {
            self.note = Some(("Set this up in the Craft launcher's Set up AI window. It will show up here once it is ready.".into(), false));
            return;
        };
        let Some(feature) = craft_ai_features::builtin().into_iter().find(|f| f.id == tool) else {
            self.note = Some(("This model is not in this version of PhotoCraft.".into(), true));
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

    fn setup_ui(&mut self, ui: &mut egui::Ui, app: &AppId, client: &AiClient, role: &Role) {
        let Some(setup) = self.setup.as_mut() else { return };
        setup.drain();
        if setup.session.busy() {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
        let Some(action) = native_model_setup(ui, &mut setup.session) else { return };
        match action {
            NativeSetupAction::Work(request) => setup.reports = Some(run(request)),
            NativeSetupAction::Use(checked) => {
                self.note = Some(match desk(app).and_then(|d| d.adopt(&checked)) {
                    Ok(adopted) if adopted.bound => ("The model is ready. Select Subject now uses it in every Craft app.".into(), false),
                    Ok(_) => ("The model is ready. Select Subject kept your earlier choice: pick the new model in the list to use it.".into(), false),
                    Err(e) => (e.message, true),
                });
                self.setup = None;
                self.refresh(client, role);
            }
            NativeSetupAction::UseSuggestedFolder => {
                let result = folders().and_then(|f| f.ensure_default().map_err(|e| e.message));
                self.review(result.map(|_| ()), client);
            }
            NativeSetupAction::FindFiles => {
                if let Some(path) = pick_folder("Find model files") {
                    let result = folders().and_then(|f| f.add(&path, None).map(|_| ()).map_err(|e| e.message));
                    self.review(result, client);
                }
            }
            NativeSetupAction::ChangeFolder => {
                if let Some(path) = pick_folder("Save model files to") {
                    let result = folders().and_then(|f| new_default(&f, &path));
                    self.review(result, client);
                }
            }
            // The session already raised the worker's cancel flag; its late answer is refused.
            NativeSetupAction::Cancel(_) => {}
            NativeSetupAction::Back => {
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
                // A different target cancels the old worker (its flag is raised by `review`).
                let _ = setup.session.review(plan);
                self.note = None;
            }
            Err(message) => self.note = Some((message, true)),
        }
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

fn shared() -> Result<SharedStore, String> {
    SharedStore::at_default_location().map_err(|e| format!("The shared AI settings could not be read: {e}"))
}

fn folders() -> Result<FolderDesk, String> {
    shared().map(FolderDesk::new)
}

/// Choices made here are the suite's: the same store the launcher writes.
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
