//! The Contextual Task Bar: a floating bar under the selection (as VectorCraft has one) with the
//! likely next steps for it. Its Generative Fill button opens the bar into a prompt, the model,
//! the quality and the resolution, so a fill is set up without leaving the canvas.
//!
//! Every picture sent is a multiple of 16 pixels on both sides, whatever the canvas or the
//! selection: image models work in 16-pixel blocks. Which qualities and sizes are offered depends
//! on the model ([`profile`]); the returned picture is stretched back over the canvas.

use super::{AiModelChoice, AiSelectStatus, FillOptions, Kind, PhotocraftApp};
use crate::canvas::ViewXform;
use crate::state::Tool;
use crate::theme::Tokens;
use egui::{Align2, Id, RichText, Stroke, Ui};
use photocraft_engine::ai_fill_cmds::round_side;
use serde_json::json;

/// How long the list of models is kept before it is read again.
const MODELS_FOR: f64 = 3.0;
/// The space between the selection and the bar, in points.
const GAP: f32 = 12.0;
/// Width of the open bar's prompt.
const PROMPT_WIDTH: f32 = 460.0;

/// The bar's state between frames.
#[derive(Default)]
pub(crate) struct Bar {
    /// Generative Fill's controls are showing.
    open: bool,
    prompt: String,
    quality: Quality,
    size: SizeChoice,
    /// The models for Generative Fill and when they were read.
    models: Option<(f64, Vec<AiModelChoice>)>,
    /// The last thing that went wrong, shown in the bar.
    error: Option<String>,
}

/// OpenAI's `quality`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Quality {
    #[default]
    Auto,
    Low,
    Medium,
    High,
}

impl Quality {
    const ALL: [Quality; 4] = [Quality::Auto, Quality::Low, Quality::Medium, Quality::High];

    fn id(self) -> &'static str {
        match self {
            Quality::Auto => "auto",
            Quality::Low => "low",
            Quality::Medium => "medium",
            Quality::High => "high",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Quality::Auto => tl!("Automatic"),
            Quality::Low => tl!("Low"),
            Quality::Medium => tl!("Medium"),
            Quality::High => tl!("High"),
        }
    }
}

/// What the person picked under Resolution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SizeChoice {
    /// The canvas's proportions at about 1.5 megapixels, or the model's closest size.
    #[default]
    Auto,
    /// The canvas's own size (rounded, and kept within what the model takes).
    Canvas,
    /// A size from the model's list.
    Preset([u32; 2]),
}

/// Which sizes a model takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sizes {
    /// Any size whose sides are multiples of 16, within [`fit_free`]'s limits (gpt-image-2).
    Free,
    /// Only these (gpt-image-1, DALL·E 2).
    Fixed(&'static [[u32; 2]]),
    /// Not known: no size is asked for, and the picture goes at the canvas's proportions.
    Unknown,
}

/// What can be asked of a model besides the prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Profile {
    pub qualities: bool,
    pub sizes: Sizes,
}

const GPT_IMAGE_1_SIZES: &[[u32; 2]] = &[[1024, 1024], [1536, 1024], [1024, 1536]];
const FREE_PRESETS: &[[u32; 2]] = &[[1024, 1024], [1536, 1024], [1024, 1536], [2048, 2048], [2048, 1152], [1152, 2048], [3840, 2160], [2160, 3840]];
const DALL_E_2_SIZES: &[[u32; 2]] = &[[256, 256], [512, 512], [1024, 1024]];
/// gpt-image-2's limits: each side at most 3840, 655,360 to 8,294,400 pixels, 3:1 at most.
const MAX_EDGE: f64 = 3840.0;
const MIN_PIXELS: f64 = 655_360.0;
const MAX_PIXELS: f64 = 8_294_400.0;
const MAX_RATIO: f64 = 3.0;
/// What Auto aims for with a model that takes any size: 1536 × 1024's pixel count.
const AUTO_PIXELS: f64 = 1_572_864.0;

/// The model's version after `gpt-image-`: `2` for `gpt-image-2-2026-04-21`, `1` for `gpt-image-1.5`.
fn gpt_image_major(model: &str) -> Option<u32> {
    let lower = model.to_ascii_lowercase();
    let rest = &lower[lower.find("gpt-image-")? + "gpt-image-".len()..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// What the model `model` (its id) takes, as OpenAI documents it.
pub(crate) fn profile(model: &str) -> Profile {
    let lower = model.to_ascii_lowercase();
    match gpt_image_major(&lower) {
        Some(major) if major >= 2 => Profile { qualities: true, sizes: Sizes::Free },
        Some(_) => Profile { qualities: true, sizes: Sizes::Fixed(GPT_IMAGE_1_SIZES) },
        None if lower.contains("chatgpt-image") => Profile { qualities: true, sizes: Sizes::Fixed(GPT_IMAGE_1_SIZES) },
        None if lower.contains("dall-e-2") => Profile { qualities: false, sizes: Sizes::Fixed(DALL_E_2_SIZES) },
        None => Profile { qualities: false, sizes: Sizes::Unknown },
    }
}

/// The sizes listed under Resolution after Auto (and Canvas, for a model that takes any size).
fn presets(sizes: Sizes) -> &'static [[u32; 2]] {
    match sizes {
        Sizes::Free => FREE_PRESETS,
        Sizes::Fixed(list) => list,
        Sizes::Unknown => &[],
    }
}

/// A size of the same proportions as `w` × `h`, with both sides multiples of 16, that a model
/// taking any size accepts: at most 3:1, each side at most 3840 and the pixel count within
/// gpt-image-2's range. `w` × `h` is kept as it is when it already fits.
pub(crate) fn fit_free(w: f64, h: f64) -> [u32; 2] {
    let (w, h) = (w.max(1.0), h.max(1.0));
    // A little inside 3:1, so rounding cannot push it past.
    let ratio = (w / h).clamp(1.0 / (MAX_RATIO - 0.02), MAX_RATIO - 0.02);
    let mut area = (w * h).clamp(MIN_PIXELS, MAX_PIXELS);
    let mut best = [1024, 1024];
    for _ in 0..400 {
        let mut fw = (area * ratio).sqrt();
        let mut fh = fw / ratio;
        let over = fw.max(fh) / MAX_EDGE;
        if over > 1.0 {
            fw /= over;
            fh /= over;
        }
        let size = [round_side(fw), round_side(fh)];
        let pixels = f64::from(size[0]) * f64::from(size[1]);
        best = size;
        if pixels < MIN_PIXELS {
            area *= 1.01;
        } else if pixels > MAX_PIXELS || f64::from(size[0].max(size[1])) > MAX_EDGE {
            area *= 0.99;
        } else {
            break;
        }
    }
    best
}

/// The listed size whose proportions are closest to `canvas`'s.
fn closest(list: &[[u32; 2]], canvas: [u32; 2]) -> Option<[u32; 2]> {
    let aspect = |s: [u32; 2]| (f64::from(s[0].max(1)) / f64::from(s[1].max(1))).ln();
    let want = aspect(canvas);
    // The largest of equally close sizes: DALL·E 2's squares all match a square canvas.
    list.iter().copied().min_by(|a, b| {
        let (da, db) = ((aspect(*a) - want).abs(), (aspect(*b) - want).abs());
        da.total_cmp(&db).then(b[0].cmp(&a[0]))
    })
}

/// The size to send for `choice` with a model of `profile`, for a canvas of `canvas` pixels: both
/// sides multiples of 16. `None` lets the picture go at the canvas's proportions with no size
/// asked of the model (the engine still rounds it to multiples of 16).
pub(crate) fn resolve(profile: Profile, choice: SizeChoice, canvas: [u32; 2]) -> Option<[u32; 2]> {
    let [cw, ch] = [f64::from(canvas[0].max(1)), f64::from(canvas[1].max(1))];
    match profile.sizes {
        Sizes::Unknown => None,
        Sizes::Fixed(list) => match choice {
            SizeChoice::Preset(p) if list.contains(&p) => Some(p),
            _ => closest(list, canvas),
        },
        Sizes::Free => Some(match choice {
            SizeChoice::Preset(p) => fit_free(f64::from(p[0]), f64::from(p[1])),
            SizeChoice::Canvas => fit_free(cw, ch),
            SizeChoice::Auto => {
                let ratio = cw / ch;
                let w = (AUTO_PIXELS * ratio).sqrt();
                fit_free(w, w / ratio)
            }
        }),
    }
}

fn size_label(s: [u32; 2]) -> String {
    format!("{} × {}", s[0], s[1])
}

fn choice_label(choice: SizeChoice) -> String {
    match choice {
        SizeChoice::Auto => tl!("Automatic").to_string(),
        SizeChoice::Canvas => tl!("Canvas size").to_string(),
        SizeChoice::Preset(p) => size_label(p),
    }
}

/// Whether `tool` makes selections: the bar shows only with these, as in Photoshop.
fn selects(tool: Tool) -> bool {
    matches!(
        tool,
        Tool::RectMarquee
            | Tool::EllipseMarquee
            | Tool::Lasso
            | Tool::PolygonLasso
            | Tool::MagneticLasso
            | Tool::MagicWand
            | Tool::QuickSelection
            | Tool::ObjectSelection
    )
}

/// What was pressed in the bar this frame; done after drawing.
enum Pressed {
    Open,
    Close,
    Menu(&'static str),
    Model(String, String),
    Generate,
    Cancel,
    ChooseModel,
}

/// Draws the bar under the selection of document `idx`, when it has one and a selection tool is
/// in use. `dragging` hides it while the person is still drawing the selection.
pub(crate) fn show(app: &mut PhotocraftApp, ctx: &egui::Context, xf: &ViewXform, idx: usize, tool: Tool, dragging: bool) {
    if app.services.ai_select.is_none() || app.session.active_index() != Some(idx) || !app.ui.dialogs.is_empty() {
        return;
    }
    let running = app.ai_select.running.as_ref().filter(|r| r.kind == Kind::Fill).map(|r| r.model.clone());
    let open = app.ai_select.bar.open;
    if running.is_none() && (!selects(tool) || dragging) && !open {
        return;
    }
    let Some(doc) = app.session.active().map(|d| d.doc.clone()) else { return };
    let Some(selection) = doc.selection.as_ref() else {
        app.ai_select.bar.open = false;
        return;
    };
    let canvas = [doc.bounds().width(), doc.bounds().height()];
    let b = app.cached_bounds(u64::MAX - 3 - doc.id.0.rotate_left(9), selection);
    if b.width() == 0 || b.height() == 0 {
        return;
    }
    let corners = [(b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1)].map(|(x, y)| xf.to_screen(x as f32, y as f32));
    let on_screen = egui::Rect::from_points(&corners);
    let area_id = Id::unique("photocraft-task-bar");
    let height = ctx.memory(|m| m.area_rect(area_id)).map_or(48.0, |r| r.height());
    // Under the selection, or over it when there is no room below.
    let (pivot, y) = if on_screen.max.y + GAP + height < xf.rect.max.y - 8.0 {
        (Align2::CENTER_TOP, on_screen.max.y + GAP)
    } else if on_screen.min.y - GAP - height > xf.rect.min.y + 8.0 {
        (Align2::CENTER_BOTTOM, on_screen.min.y - GAP)
    } else {
        (Align2::CENTER_BOTTOM, xf.rect.max.y - 16.0)
    };
    let x = on_screen.center().x.clamp(xf.rect.min.x + 40.0, xf.rect.max.x - 40.0);

    if app.ai_select.bar.models.as_ref().is_none_or(|(at, _)| ctx.input(|i| i.time) - at > MODELS_FOR)
        && let Some(service) = app.services.ai_select.as_ref()
    {
        app.ai_select.bar.models = Some((ctx.input(|i| i.time), service.fill_models()));
    }
    let status = app.services.ai_select.as_ref().map(|s| s.fill_status());
    let t = Tokens::get(ctx);
    let mut pressed = None;
    egui::Area::new(area_id).order(egui::Order::Foreground).pivot(pivot).fixed_pos(egui::pos2(x, y)).constrain_to(xf.rect.shrink(4.0)).show(ctx, |ui| {
        egui::Frame::new()
            .fill(t.card)
            .stroke(Stroke::new(1.0, t.card_border))
            .corner_radius(t.radius_lg.max(8.0))
            .inner_margin(egui::Margin::symmetric(8, 6))
            .shadow(egui::epaint::Shadow { offset: [0, 4], blur: 14, spread: 0, color: t.shadow })
            .show(ui, |ui| {
                pressed = if let Some(model) = &running {
                    running_row(ui, model)
                } else if open {
                    fill_panel(ui, &mut app.ai_select.bar, status.as_ref(), canvas, &t)
                } else {
                    actions_row(ui)
                };
            });
    });
    let Some(pressed) = pressed else { return };
    apply(app, ctx, pressed, canvas);
}

fn actions_row(ui: &mut Ui) -> Option<Pressed> {
    let mut pressed = None;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        if ui.button(tl!("Generative Fill")).clicked() {
            pressed = Some(Pressed::Open);
        }
        ui.separator();
        if ui.button(tl!("Invert")).on_hover_text(tl!("Inverse Selection")).clicked() {
            pressed = Some(Pressed::Menu("select.inverse"));
        }
        if ui.button(tl!("Deselect")).clicked() {
            pressed = Some(Pressed::Menu("select.deselect"));
        }
    });
    pressed
}

fn running_row(ui: &mut Ui, model: &str) -> Option<Pressed> {
    let mut pressed = None;
    ui.horizontal(|ui| {
        ui.spinner();
        ui.label(crate::i18n::fmt(tl!("Generating with {model}…"), &[("model", model)]));
        if ui.button(tl!("Cancel")).clicked() {
            pressed = Some(Pressed::Cancel);
        }
    });
    ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
    pressed
}

fn fill_panel(ui: &mut Ui, bar: &mut Bar, status: Option<&AiSelectStatus>, canvas: [u32; 2], t: &Tokens) -> Option<Pressed> {
    let mut pressed = None;
    ui.set_max_width(PROMPT_WIDTH + 24.0);
    ui.horizontal(|ui| {
        ui.label(RichText::new(tl!("Generative Fill")).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("×").on_hover_text(tl!("Close")).clicked() {
                pressed = Some(Pressed::Close);
            }
        });
    });
    let models = bar.models.as_ref().map(|(_, m)| m.clone()).unwrap_or_default();
    let current = models.iter().find(|m| m.selected).cloned();
    if models.is_empty() || !matches!(status, Some(AiSelectStatus::Ready { .. })) {
        let reason = match status {
            Some(AiSelectStatus::Unavailable { reason }) => reason.clone(),
            _ => tl!("Generative Fill needs an AI image model").to_string(),
        };
        ui.label(RichText::new(reason).color(t.text_dim));
        if ui.button(tl!("Choose a model…")).clicked() {
            pressed = Some(Pressed::ChooseModel);
        }
        return pressed;
    }
    ui.add_space(2.0);
    let prompt = ui.add(
        egui::TextEdit::multiline(&mut bar.prompt)
            .id_salt("photocraft-task-bar-prompt")
            .hint_text(tl!("Describe what should appear in the selection (leave it empty to match its surroundings)"))
            .desired_rows(3)
            .desired_width(PROMPT_WIDTH),
    );
    if prompt.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.command) {
        pressed = Some(Pressed::Generate);
    }
    let profile = current.as_ref().map_or(Profile { qualities: false, sizes: Sizes::Unknown }, |m| profile(&m.model));
    ui.add_space(4.0);
    ui.horizontal_wrapped(|ui| {
        ui.label(tl!("Model"));
        // Model names are shown as the service gives them, not translated.
        let mut picked = models.iter().position(|m| m.selected).unwrap_or(usize::MAX);
        let options: Vec<(usize, &str)> = models.iter().enumerate().map(|(i, m)| (i, m.label.as_str())).collect();
        if crate::widgets::dropdown_literal(ui, "photocraft-task-bar-model", tl!("Model"), &mut picked, &options, 190.0)
            && let Some(m) = models.get(picked)
            && !m.selected
        {
            pressed = Some(Pressed::Model(m.connection.clone(), m.model.clone()));
        }
        if profile.qualities {
            ui.add_space(6.0);
            ui.label(tl!("Quality"));
            let options: Vec<(Quality, &str)> = Quality::ALL.iter().map(|q| (*q, q.label())).collect();
            crate::widgets::dropdown_literal(ui, "photocraft-task-bar-quality", tl!("Quality"), &mut bar.quality, &options, 110.0);
        }
        if profile.sizes != Sizes::Unknown {
            ui.add_space(6.0);
            ui.label(tl!("Resolution"));
            let mut choices = vec![SizeChoice::Auto];
            if profile.sizes == Sizes::Free {
                choices.push(SizeChoice::Canvas);
            }
            choices.extend(presets(profile.sizes).iter().map(|p| SizeChoice::Preset(*p)));
            if !choices.contains(&bar.size) {
                bar.size = SizeChoice::Auto;
            }
            let labels: Vec<(SizeChoice, String)> = choices.iter().map(|c| (*c, choice_label(*c))).collect();
            let options: Vec<(SizeChoice, &str)> = labels.iter().map(|(c, l)| (*c, l.as_str())).collect();
            crate::widgets::dropdown_literal(ui, "photocraft-task-bar-size", tl!("Resolution"), &mut bar.size, &options, 150.0);
        }
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let sent = resolve(profile, bar.size, canvas);
        let note = match sent {
            Some(s) => crate::i18n::fmt(tl!("Sent at {size} px (multiples of 16)"), &[("size", &size_label(s))]),
            None => tl!("Sent at the canvas's proportions, rounded to multiples of 16").to_string(),
        };
        ui.label(RichText::new(note).small().color(t.text_dim));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let generate = egui::Button::new(RichText::new(tl!("Generate")).color(t.primary_text)).fill(t.primary_bg);
            if ui.add(generate).on_hover_text(tl!("Ctrl+Enter")).clicked() {
                pressed = Some(Pressed::Generate);
            }
        });
    });
    if let Some(error) = &bar.error {
        ui.colored_label(t.danger, error);
    }
    pressed
}

fn apply(app: &mut PhotocraftApp, ctx: &egui::Context, pressed: Pressed, canvas: [u32; 2]) {
    let bar = &mut app.ai_select.bar;
    match pressed {
        Pressed::Open => {
            bar.open = true;
            bar.error = None;
            bar.models = None;
        }
        Pressed::Close => {
            bar.open = false;
            bar.error = None;
        }
        Pressed::Menu(id) => {
            if let Err(e) = crate::menus::invoke(app, ctx, id, json!({})) {
                app.ui.status = e;
                app.ui.status_error = true;
            }
        }
        Pressed::ChooseModel => {
            crate::prefs_ui::open_preferences(app, "integrations");
        }
        Pressed::Model(connection, model) => {
            let chosen = app.services.ai_select.as_ref().map(|s| s.choose_fill_model(&connection, &model));
            app.ai_select.bar.models = None;
            if let Some(Err(e)) = chosen {
                app.ai_select.bar.error = Some(e);
            } else {
                app.ai_select.bar.error = None;
            }
        }
        Pressed::Cancel => {
            if let Some(running) = app.ai_select.running.as_mut() {
                running.job.cancel();
            }
        }
        Pressed::Generate => {
            let current = bar.models.as_ref().and_then(|(_, m)| m.iter().find(|m| m.selected).cloned());
            let profile = current.as_ref().map_or(Profile { qualities: false, sizes: Sizes::Unknown }, |m| profile(&m.model));
            let options = FillOptions { quality: profile.qualities.then(|| bar.quality.id().to_string()), size: resolve(profile, bar.size, canvas) };
            let prompt = bar.prompt.clone();
            match super::start_fill(app, &prompt, options) {
                Ok(_) => {
                    let bar = &mut app.ai_select.bar;
                    bar.error = None;
                    // A consent question opens as a dialog; the bar comes back after it.
                    bar.open = app.ai_select.awaiting.is_some();
                }
                Err(e) => app.ai_select.bar.error = Some(e),
            }
        }
    }
    ctx.request_repaint();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_get_the_options_openai_documents() {
        assert_eq!(profile("gpt-image-2"), Profile { qualities: true, sizes: Sizes::Free });
        assert_eq!(profile("gpt-image-2.5-flare-2026-09-08"), Profile { qualities: true, sizes: Sizes::Free });
        assert_eq!(profile("gpt-image-1.5"), Profile { qualities: true, sizes: Sizes::Fixed(GPT_IMAGE_1_SIZES) });
        assert_eq!(profile("gpt-image-1-mini"), Profile { qualities: true, sizes: Sizes::Fixed(GPT_IMAGE_1_SIZES) });
        assert_eq!(profile("chatgpt-image-latest"), Profile { qualities: true, sizes: Sizes::Fixed(GPT_IMAGE_1_SIZES) });
        assert_eq!(profile("dall-e-2"), Profile { qualities: false, sizes: Sizes::Fixed(DALL_E_2_SIZES) });
        assert_eq!(profile("flux-fill"), Profile { qualities: false, sizes: Sizes::Unknown });
    }

    fn multiple_of_16(s: [u32; 2]) -> bool {
        s[0] % 16 == 0 && s[1] % 16 == 0
    }

    #[test]
    fn every_size_sent_is_a_multiple_of_16_and_fits_the_model() {
        let free = profile("gpt-image-2");
        let fixed = profile("gpt-image-1");
        for canvas in [[1, 1], [37, 1000], [801, 600], [1920, 1080], [10_000, 333], [4000, 4000], [123, 45_678]] {
            for choice in [SizeChoice::Auto, SizeChoice::Canvas, SizeChoice::Preset([1536, 1024]), SizeChoice::Preset([3840, 2160])] {
                let s = resolve(free, choice, canvas).unwrap();
                assert!(multiple_of_16(s), "{canvas:?} {choice:?} → {s:?}");
                let pixels = f64::from(s[0]) * f64::from(s[1]);
                assert!((MIN_PIXELS..=MAX_PIXELS).contains(&pixels), "{canvas:?} {choice:?} → {s:?}");
                assert!(s[0].max(s[1]) <= 3840 && f64::from(s[0].max(s[1])) / f64::from(s[0].min(s[1])) <= MAX_RATIO, "{s:?}");
                let f = resolve(fixed, choice, canvas).unwrap();
                assert!(GPT_IMAGE_1_SIZES.contains(&f), "{f:?}");
            }
        }
    }

    #[test]
    fn auto_follows_the_canvas_proportions() {
        let free = profile("gpt-image-2");
        let s = resolve(free, SizeChoice::Auto, [1920, 1080]).unwrap();
        let ratio = f64::from(s[0]) / f64::from(s[1]);
        assert!((ratio - 16.0 / 9.0).abs() < 0.02, "{s:?}");
        assert_eq!(resolve(free, SizeChoice::Canvas, [1920, 1080]), Some([1920, 1088]), "rounded, not scaled");
        assert_eq!(resolve(free, SizeChoice::Preset([2048, 1152]), [10, 10]), Some([2048, 1152]));
        let fixed = profile("gpt-image-1");
        assert_eq!(resolve(fixed, SizeChoice::Auto, [1920, 1080]), Some([1536, 1024]));
        assert_eq!(resolve(fixed, SizeChoice::Auto, [600, 900]), Some([1024, 1536]));
        assert_eq!(resolve(fixed, SizeChoice::Auto, [500, 520]), Some([1024, 1024]));
        assert_eq!(resolve(fixed, SizeChoice::Preset([1024, 1536]), [1920, 1080]), Some([1024, 1536]));
        assert_eq!(resolve(profile("dall-e-2"), SizeChoice::Auto, [300, 300]), Some([1024, 1024]));
        assert_eq!(resolve(profile("other"), SizeChoice::Auto, [300, 300]), None);
    }
}
