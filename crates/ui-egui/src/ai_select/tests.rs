use super::*;
use std::cell::RefCell;
use std::rc::Rc;

/// What the scripted service saw and how it answers.
#[derive(Default)]
struct Script {
    ready: bool,
    online: bool,
    starts: Vec<(AiSelectRequest, Option<AiConsentTerm>)>,
    /// Polls answered `Running` before the job finishes.
    running_polls: usize,
    matched: bool,
    cancelled: bool,
    /// Frames Preferences drew the service's settings in.
    settings_drawn: usize,
}

struct FakeService(Rc<RefCell<Script>>);
struct FakeJob(Rc<RefCell<Script>>, usize);

impl AiSelect for FakeService {
    fn status(&self) -> AiSelectStatus {
        if self.0.borrow().ready {
            AiSelectStatus::Ready { model: "Test SAM".into() }
        } else {
            AiSelectStatus::Unavailable { reason: "No model chosen".into() }
        }
    }
    fn start(&self, request: AiSelectRequest, consent: Option<AiConsentTerm>) -> Result<AiSelectStart, String> {
        let mut s = self.0.borrow_mut();
        s.starts.push((request, consent));
        if s.online && consent.is_none() {
            return Ok(AiSelectStart::NeedsConsent(AiConsentAsk { destination: "https://example.test".into(), items: vec!["Picture, 1 KB".into()] }));
        }
        Ok(AiSelectStart::Started(Box::new(FakeJob(self.0.clone(), 0))))
    }
    fn preferences_ui(&mut self, ui: &mut egui::Ui) {
        self.0.borrow_mut().settings_drawn += 1;
        ui.label("Fake model picker");
    }
}

impl AiSelectJob for FakeJob {
    fn poll(&mut self) -> AiJobPoll {
        let s = self.0.borrow();
        if s.cancelled {
            return AiJobPoll::Cancelled;
        }
        if self.1 < s.running_polls {
            self.1 += 1;
            return AiJobPoll::Running(None);
        }
        // A mask selecting the left half of a 4 × 4 picture.
        let mut image = photocraft_codecs::Image::new(4, 4, photocraft_codecs::ChannelLayout::Gray, photocraft_codecs::SampleType::U8).unwrap();
        for (i, v) in image.data_mut().iter_mut().enumerate() {
            *v = if s.matched && i % 4 < 2 { 255 } else { 0 };
        }
        let mask_png = photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &Default::default()).unwrap();
        AiJobPoll::Done { mask_png, matched: s.matched, name: None }
    }
    fn cancel(&mut self) {
        self.0.borrow_mut().cancelled = true;
    }
}

fn make_app(script: Script) -> (PhotocraftApp, Rc<RefCell<Script>>) {
    let script = Rc::new(RefCell::new(script));
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 40, "height": 20})).unwrap();
    let services = crate::Services { ai_select: Some(Box::new(FakeService(script.clone()))), ..Default::default() };
    (PhotocraftApp::new(s, services), script)
}

fn tick(app: &mut PhotocraftApp) {
    let ctx = egui::Context::default();
    super::tick(app, &ctx);
}

fn coverage(app: &PhotocraftApp, x: i32, y: i32) -> f32 {
    app.session.active().unwrap().doc.selection.as_ref().map_or(0.0, |m| m.sample_channel(x, y, 0))
}

#[test]
fn no_model_ready_keeps_the_built_in_command() {
    let (mut app, script) = make_app(Script::default());
    app.run("select.subject", json!({})).unwrap();
    assert!(script.borrow().starts.is_empty());
    assert!(!busy(&app));
}

#[test]
fn a_ready_local_model_selects_with_its_mask_as_one_undo_step() {
    let (mut app, script) = make_app(Script { ready: true, running_polls: 2, matched: true, ..Default::default() });
    let before = app.session.active().unwrap().history.past_len();
    let v = app.run("select.subject", json!({})).unwrap();
    assert_eq!(v["pending"], json!(true));
    {
        let s = script.borrow();
        let (request, consent) = &s.starts[0];
        assert!(consent.is_none());
        assert_eq!(request.prompt, PROMPT);
        let picture = photocraft_codecs::decode(&request.png).unwrap();
        assert_eq!((picture.width(), picture.height()), (40, 20));
    }
    tick(&mut app);
    tick(&mut app);
    assert!(busy(&app), "still running");
    assert!(app.session.active().unwrap().doc.selection.is_none());
    tick(&mut app);
    assert!(!busy(&app));
    // The 4 × 4 mask's left half, stretched over the 40 × 20 canvas.
    assert!(coverage(&app, 5, 10) > 0.9);
    assert!(coverage(&app, 35, 10) < 0.1);
    assert_eq!(app.session.active().unwrap().history.past_len(), before + 1);
    assert!(app.ui.status.contains("Test SAM"), "{}", app.ui.status);
    app.session.undo();
    assert!(app.session.active().unwrap().doc.selection.is_none());
}

#[test]
fn an_online_model_asks_first_and_sends_only_after_agreement() {
    let (mut app, script) = make_app(Script { ready: true, online: true, matched: true, ..Default::default() });
    let v = app.run("select.subject", json!({})).unwrap();
    assert_eq!(v["consentPending"], json!(true));
    assert!(!busy(&app));
    let dialog = v["dialog"].as_u64().unwrap();
    assert!(owns(&app.ui.dialog_mut(dialog).unwrap().fields));
    app.ui.dialog_mut(dialog).unwrap().fields.insert("term".into(), json!("one_hour"));
    crate::dialogs::confirm(&mut app, dialog).unwrap();
    assert_eq!(script.borrow().starts.last().unwrap().1, Some(AiConsentTerm::OneHour));
    assert!(busy(&app));
    tick(&mut app);
    assert!(coverage(&app, 5, 10) > 0.9);
}

#[test]
fn nothing_found_and_cancel_leave_the_selection_alone() {
    let (mut app, _) = make_app(Script { ready: true, matched: false, ..Default::default() });
    app.run("select.subject", json!({})).unwrap();
    tick(&mut app);
    assert!(app.session.active().unwrap().doc.selection.is_none());
    assert!(app.ui.status.contains("No subject found"), "{}", app.ui.status);

    let (mut app, script) = make_app(Script { ready: true, running_polls: 100, matched: true, ..Default::default() });
    app.run("select.subject", json!({})).unwrap();
    script.borrow_mut().cancelled = true;
    tick(&mut app);
    assert!(!busy(&app));
    assert!(app.session.active().unwrap().doc.selection.is_none());
}

#[test]
fn automation_keeps_the_built_in_command() {
    let (mut app, script) = make_app(Script { ready: true, matched: true, ..Default::default() });
    app.run("select.subject", json!({"builtIn": true})).unwrap();
    assert!(script.borrow().starts.is_empty());
}

#[test]
fn preferences_integrations_shows_the_services_model_settings() {
    let (mut app, script) = make_app(Script::default());
    crate::prefs_ui::open_preferences(&mut app, "integrations");
    let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("semibold".into()))) {
                crate::dialogs::show(app, ui.ctx());
            }
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(3);
    assert!(script.borrow().settings_drawn > 0);
    use egui_kittest::kittest::Queryable as _;
    h.get_by_label("Fake model picker");
    h.get_by_label("AI models");

    // Other sections leave it out.
    let general = {
        let app = h.state_mut();
        let id = app.ui.dialogs.last().map(|d| d.id).unwrap();
        app.ui.dialog_mut(id).unwrap().fields.insert("section".into(), json!("general"));
        script.borrow().settings_drawn
    };
    h.run_steps(3);
    assert_eq!(script.borrow().settings_drawn, general);
}
