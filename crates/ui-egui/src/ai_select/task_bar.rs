//! Generative Fill in the Contextual Task Bar (`widgets::task_bar`, shared with the Crop tool):
//! under the selection, or under the canvas when nothing is selected (the fill then works on the
//! whole canvas), with the likely next steps. Its Generative Fill button opens the bar into a
//! panel with the prompt, reference pictures, the model, the quality, the resolution and the
//! likely cost, so a fill is set up without leaving the canvas.
//!
//! Every picture sent is a multiple of 16 pixels on both sides, whatever the canvas or the
//! selection: image models work in 16-pixel blocks. Which qualities and sizes are offered depends
//! on the model ([`profile`]); the returned picture is stretched back over the canvas.
//!
//! The open bar also takes up to three reference pictures, each with an `@tag` the description
//! can name (typing `@` offers them), and shows what one generation is likely to cost
//! ([`super::cost`]).

use super::references::{self, Reference};
use super::{AiModelChoice, AiSelectStatus, FillOptions, Kind, PhotocraftApp, cost};
use crate::canvas::ViewXform;
use crate::state::Tool;
use crate::theme::Tokens;
use crate::widgets::task_bar;
use egui::{Align2, RichText, Stroke, Ui};
use photocraft_engine::ai_fill_cmds::{fill_region, round_side};
use serde_json::json;

/// How long the list of models is kept before it is read again.
const MODELS_FOR: f64 = 3.0;
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
    /// Reference pictures, at most [`references::MAX`].
    pub(crate) references: Vec<Reference>,
    /// Where to put the prompt's cursor next frame (chars), after a tag was completed.
    cursor_to: Option<usize>,
    /// Output tokens per 1024 × 1024 that each (model, quality) really used, from the last
    /// generation: better than the published figures for models OpenAI documents loosely.
    learned: std::collections::HashMap<(String, String), f64>,
    /// What the last generation cost, in US dollars.
    pub(crate) last_cost: Option<f64>,
}

impl Bar {
    /// Remembers that `model` at `quality` made `output` tokens for a picture of `pixels`.
    pub(crate) fn learn(&mut self, model: &str, quality: Option<&str>, pixels: f64, output: u64) {
        if pixels > 0.0 && output > 0 {
            let per = output as f64 * (1024.0 * 1024.0) / pixels;
            self.learned.insert((model.to_owned(), quality.unwrap_or("auto").to_owned()), per);
        }
    }
}

/// OpenAI's `quality`. The base GPT Image models take Auto to High; gpt-image-2.5 and later
/// (Sunburst, Flare) also take Extra High and Max.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Quality {
    #[default]
    Auto,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

/// The qualities of the base GPT Image models.
const BASE_QUALITIES: &[Quality] = &[Quality::Auto, Quality::Low, Quality::Medium, Quality::High];
/// gpt-image-2.5 and later.
const ALL_QUALITIES: &[Quality] = &[Quality::Auto, Quality::Low, Quality::Medium, Quality::High, Quality::XHigh, Quality::Max];

impl Quality {
    fn id(self) -> &'static str {
        match self {
            Quality::Auto => "auto",
            Quality::Low => "low",
            Quality::Medium => "medium",
            Quality::High => "high",
            Quality::XHigh => "xhigh",
            Quality::Max => "max",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Quality::Auto => tl!("Automatic"),
            Quality::Low => tl!("Low"),
            Quality::Medium => tl!("Medium"),
            Quality::High => tl!("High"),
            Quality::XHigh => tl!("Extra high"),
            Quality::Max => tl!("Maximum"),
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
    /// The qualities it takes; none when it takes no `quality`.
    pub qualities: &'static [Quality],
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

/// The model's version after `gpt-image-`: `(2, 0)` for `gpt-image-2-2026-04-21`, `(1, 5)` for
/// `gpt-image-1.5`, `(2, 5)` for `gpt-image-2.5-flare`.
pub(crate) fn gpt_image_version(model: &str) -> Option<(u32, u32)> {
    let lower = model.to_ascii_lowercase();
    let rest = &lower[lower.find("gpt-image-")? + "gpt-image-".len()..];
    let major: String = rest.chars().take_while(char::is_ascii_digit).collect();
    let after = &rest[major.len()..];
    let minor: String = after.strip_prefix('.').map(|m| m.chars().take_while(char::is_ascii_digit).collect()).unwrap_or_default();
    Some((major.parse().ok()?, minor.parse().unwrap_or(0)))
}

/// What the model `model` (its id) takes, as OpenAI documents it.
pub(crate) fn profile(model: &str) -> Profile {
    let lower = model.to_ascii_lowercase();
    match gpt_image_version(&lower) {
        Some(v) if v >= (2, 5) => Profile { qualities: ALL_QUALITIES, sizes: Sizes::Free },
        Some(v) if v >= (2, 0) => Profile { qualities: BASE_QUALITIES, sizes: Sizes::Free },
        Some(_) => Profile { qualities: BASE_QUALITIES, sizes: Sizes::Fixed(GPT_IMAGE_1_SIZES) },
        None if lower.contains("chatgpt-image") => Profile { qualities: BASE_QUALITIES, sizes: Sizes::Fixed(GPT_IMAGE_1_SIZES) },
        None if lower.contains("dall-e-2") => Profile { qualities: &[], sizes: Sizes::Fixed(DALL_E_2_SIZES) },
        None => Profile { qualities: &[], sizes: Sizes::Unknown },
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

fn choice_label(choice: SizeChoice, target: Target) -> String {
    match choice {
        SizeChoice::Auto => tl!("Automatic").to_string(),
        SizeChoice::Canvas if target.selection => tl!("Area size").to_string(),
        SizeChoice::Canvas => tl!("Canvas size").to_string(),
        SizeChoice::Preset(p) => size_label(p),
    }
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
    AddReference,
    RemoveReference(usize),
    /// Put `@tag` into the prompt: at the partial tag `(at, typed chars)`, or at the end.
    Tag(String, Option<(usize, usize)>),
}

/// What Generative Fill works on: the selection with some picture around it, or the whole canvas.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Target {
    /// The canvas part sent ([`fill_region`]), in canvas pixels.
    pub region: photocraft_geom::Rect,
    /// There is a selection (else the whole canvas is painted).
    pub selection: bool,
}

impl Target {
    pub(crate) fn size(self) -> [u32; 2] {
        [self.region.width(), self.region.height()]
    }
}

/// The bar shows: an AI service, the Window › Contextual Task Bar toggle on, no dialog over the
/// canvas, the active document, and not the Crop tool (which has its own bar).
fn wanted(app: &PhotocraftApp, idx: usize, tool: Tool) -> bool {
    app.services.ai_select.is_some()
        && app.ui.panels.context_task_bar
        && app.ui.dialogs.is_empty()
        && app.session.active_index() == Some(idx)
        && tool != Tool::Crop
}

/// Draws the bar under the selection of document `idx`, or under the canvas when nothing is
/// selected (Generative Fill then works on the whole canvas). `dragging` hides the closed bar
/// while the person is still drawing.
pub(crate) fn show(app: &mut PhotocraftApp, ctx: &egui::Context, xf: &ViewXform, idx: usize, tool: Tool, dragging: bool) {
    if !wanted(app, idx, tool) {
        return;
    }
    let running = app.ai_select.running.as_ref().filter(|r| r.kind == Kind::Fill).map(|r| r.model.clone());
    let open = app.ai_select.bar.open;
    if running.is_none() && dragging && !open {
        return;
    }
    let Some(doc) = app.session.active().map(|d| d.doc.clone()) else { return };
    let canvas = doc.bounds();
    if canvas.is_empty() {
        return;
    }
    let selected = match doc.selection.as_ref() {
        Some(selection) => Some(app.cached_bounds(u64::MAX - 3 - doc.id.0.rotate_left(9), selection)).filter(|b| !b.is_empty()),
        None => None,
    };
    let target = Target { region: fill_region(canvas, selected), selection: selected.is_some() };
    let anchor_px = selected.unwrap_or(canvas);
    let corners = [(anchor_px.x0, anchor_px.y0), (anchor_px.x1, anchor_px.y1)].map(|(x, y)| xf.to_screen(x as f32, y as f32));
    let anchor = egui::Rect::from_two_pos(corners[0], corners[1]);

    if app.ai_select.bar.models.as_ref().is_none_or(|(at, _)| ctx.input(|i| i.time) - at > MODELS_FOR)
        && let Some(service) = app.services.ai_select.as_ref()
    {
        app.ai_select.bar.models = Some((ctx.input(|i| i.time), service.fill_models()));
    }
    let status = app.services.ai_select.as_ref().map(|s| s.fill_status());
    let t = Tokens::get(ctx);
    let mut place = app.ui.panels.task_bar;
    // The bar sits inside the view; an anchor off screen still leaves it at the nearest edge.
    let area = xf.rect.shrink(4.0);
    let bar = &mut app.ai_select.bar;
    let shown = task_bar::show(ctx, &mut place, anchor, area, |ui, more| {
        if let Some(model) = &running {
            let p = running_row(ui, model);
            more.show(ui);
            p
        } else if open {
            fill_panel(ui, bar, status.as_ref(), target, &t, more)
        } else {
            actions_row(ui, target, &t, more)
        }
    });
    app.ui.panels.task_bar = place;
    let Some(shown) = shown else { return };
    if shown.hide {
        app.ui.panels.context_task_bar = false;
        return;
    }
    let r = shown.rect;
    app.ui.task_bar_shown = Some([r.min.x, r.min.y, r.max.x, r.max.y]);
    if let Some(pressed) = shown.inner {
        apply(app, ctx, pressed, target);
    }
}

/// A bar button with an icon before its words.
fn icon_button(ui: &mut Ui, icon: &str, label: &str, t: &Tokens) -> egui::Response {
    let mut atoms = egui::Atoms::default();
    atoms.push_right(crate::icons::image(icon, 14.0, t.icon));
    atoms.push_right(label.to_owned());
    ui.add(egui::Button::new(atoms).min_size(egui::vec2(0.0, task_bar::BUTTON)))
}

fn actions_row(ui: &mut Ui, target: Target, t: &Tokens, more: &mut task_bar::MoreMenu) -> Option<Pressed> {
    let mut pressed = None;
    if icon_button(ui, "sparkles", tl!("Generative Fill"), t).clicked() {
        pressed = Some(Pressed::Open);
    }
    ui.separator();
    if target.selection {
        if icon_button(ui, "squares-subtract", tl!("Invert"), t).clicked() {
            pressed = Some(Pressed::Menu("select.inverse"));
        }
        if icon_button(ui, "x", tl!("Deselect"), t).clicked() {
            pressed = Some(Pressed::Menu("select.deselect"));
        }
    } else if icon_button(ui, "wand-sparkles", tl!("Select Subject"), t).clicked() {
        pressed = Some(Pressed::Menu("select.subject"));
    }
    more.show(ui);
    pressed
}

fn running_row(ui: &mut Ui, model: &str) -> Option<Pressed> {
    let mut pressed = None;
    ui.add_space(4.0);
    ui.spinner();
    ui.label(crate::i18n::fmt(tl!("Generating with {model}…"), &[("model", model)]));
    if ui.button(tl!("Cancel")).clicked() {
        pressed = Some(Pressed::Cancel);
    }
    ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
    pressed
}

/// A small rounded label: what the fill works on.
fn chip(ui: &mut Ui, icon: &str, text: &str, t: &Tokens) {
    egui::Frame::new().fill(t.accent_soft).corner_radius(10.0).inner_margin(egui::Margin::symmetric(8, 2)).show(ui, |ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.add(crate::icons::image(icon, 12.0, t.accent));
        ui.label(RichText::new(text).small().color(t.accent));
    });
}

/// A titled group of controls in the open panel.
fn section(ui: &mut Ui, title: &str, t: &Tokens) {
    ui.add_space(6.0);
    ui.label(RichText::new(title).small().color(t.text_dim));
}

fn fill_panel(ui: &mut Ui, bar: &mut Bar, status: Option<&AiSelectStatus>, target: Target, t: &Tokens, more: &mut task_bar::MoreMenu) -> Option<Pressed> {
    let mut pressed = None;
    ui.vertical(|ui| {
        ui.set_width(PROMPT_WIDTH);
        ui.spacing_mut().item_spacing.y = 4.0;
        // Header: title, what it works on, ⋯ and close.
        ui.horizontal(|ui| {
            ui.add(crate::icons::image("sparkles", 16.0, t.accent));
            ui.label(RichText::new(tl!("Generative Fill")).strong());
            if target.selection {
                chip(ui, "square-dashed", tl!("Selection"), t);
            } else {
                chip(ui, "scan", tl!("Whole canvas"), t);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if crate::icons::button_with_icon_size_tip(ui, "x", task_bar::BUTTON, 14.0, false, tl!("Close"), false).clicked() {
                    pressed = Some(Pressed::Close);
                }
                more.show(ui);
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
            return;
        }
        if let Some(p) = prompt_box(ui, bar, target, t) {
            pressed = Some(p);
        }
        let profile = current.as_ref().map_or(Profile { qualities: &[], sizes: Sizes::Unknown }, |m| profile(&m.model));
        let takes = current.as_ref().is_some_and(|m| takes_references(&m.model));
        section(ui, tl!("References"), t);
        if let Some(p) = references_row(ui, bar, takes, t) {
            pressed = Some(p);
        }
        section(ui, tl!("Settings"), t);
        if let Some(p) = options_row(ui, bar, &models, profile, target) {
            pressed = Some(p);
        }
        ui.add_space(6.0);
        ui.separator();
        ui.horizontal(|ui| {
            let sent = resolve(profile, bar.size, target.size());
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                if let Some(model) = &current {
                    let (text, tip) = cost_line(bar, model, profile, sent, target);
                    ui.label(RichText::new(text).strong().color(t.text)).on_hover_text(tip);
                }
                if let Some(usd) = bar.last_cost {
                    ui.label(RichText::new(crate::i18n::fmt(tl!("Last generation cost about {cost}"), &[("cost", &cost::dollars_label(usd))])).small().color(t.text_dim));
                }
                let note = match sent {
                    Some(s) => crate::i18n::fmt(tl!("Sent at {size} px (multiples of 16)"), &[("size", &size_label(s))]),
                    None => tl!("Sent at the canvas's proportions, rounded to multiples of 16").to_string(),
                };
                ui.label(RichText::new(note).small().color(t.text_dim));
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let mut atoms = egui::Atoms::default();
                atoms.push_right(crate::icons::image("sparkles", 14.0, t.primary_text));
                atoms.push_right(RichText::new(tl!("Generate")).strong().color(t.primary_text));
                let generate = egui::Button::new(atoms).fill(t.primary_bg).min_size(egui::vec2(110.0, 30.0));
                if ui.add(generate).on_hover_text(tl!("Ctrl+Enter")).clicked() {
                    pressed = Some(Pressed::Generate);
                }
            });
        });
        if let Some(error) = &bar.error {
            ui.colored_label(t.danger, error);
        }
    });
    pressed
}

/// The description, with `@` completion and the attached tags highlighted.
fn prompt_box(ui: &mut Ui, bar: &mut Bar, target: Target, t: &Tokens) -> Option<Pressed> {
    let mut pressed = None;
    let tags: Vec<String> = bar.references.iter().map(|r| r.tag.clone()).collect();
    let (accent, soft) = (t.accent, t.accent_soft);
    let mut layouter = |ui: &Ui, text: &dyn egui::TextBuffer, wrap: f32| ui.fonts_mut(|f| f.layout_job(tagged_job(ui, text.as_str(), &tags, accent, soft, wrap)));
    let hint = if target.selection {
        tl!("Describe what should appear in the selection (leave it empty to match its surroundings). Type @ to name a reference.")
    } else {
        tl!("Describe what to create or change on the whole canvas. Type @ to name a reference.")
    };
    let mut output = egui::TextEdit::multiline(&mut bar.prompt)
        .id_salt("photocraft-task-bar-prompt")
        .hint_text(hint)
        .desired_rows(3)
        .desired_width(f32::INFINITY)
        .margin(egui::Margin::symmetric(8, 6))
        .layouter(&mut layouter)
        .show(ui);
    if let Some(at) = bar.cursor_to.take() {
        let range = egui::text::CCursorRange::one(egui::text::CCursor::new(at));
        output.state.cursor.set_char_range(Some(range));
        output.state.store(ui.ctx(), output.response.id);
        output.response.request_focus();
    }
    let prompt = &output.response;
    // Typing `@`: the tags that match; Tab or a click completes the first / the one clicked.
    let typing = output.cursor_range.filter(|_| prompt.has_focus()).and_then(|r| references::typing_tag(&bar.prompt, r.primary.index.0));
    let mut tab_taken = false;
    if let Some((at, typed)) = &typing {
        let matching: Vec<&String> = tags.iter().filter(|tag| tag.starts_with(&typed.to_lowercase())).collect();
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            if tags.is_empty() {
                ui.label(RichText::new(tl!("Attach a reference to name it with @")).small().color(t.text_dim));
            }
            for (i, tag) in matching.iter().enumerate() {
                let chip = ui.add(egui::Button::new(RichText::new(format!("@{tag}")).color(t.accent)).small());
                let tab = i == 0 && ui.input(|inp| inp.key_pressed(egui::Key::Tab));
                if chip.clicked() || tab {
                    tab_taken |= tab;
                    pressed = Some(Pressed::Tag((*tag).clone(), Some((*at, typed.chars().count()))));
                }
            }
        });
    }
    if tab_taken {
        ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Tab));
    }
    if prompt.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.command) {
        pressed = Some(Pressed::Generate);
    }
    let unknown = references::unknown_tags(&bar.prompt, &tags);
    if !unknown.is_empty() {
        let names: Vec<String> = unknown.iter().map(|u| format!("@{u}")).collect();
        ui.label(RichText::new(crate::i18n::fmt(tl!("No reference is called {tags}"), &[("tags", &names.join(", "))])).small().color(t.warning));
    }
    pressed
}

/// Model, quality and resolution.
fn options_row(ui: &mut Ui, bar: &mut Bar, models: &[AiModelChoice], profile: Profile, target: Target) -> Option<Pressed> {
    let mut pressed = None;
    egui::Grid::new("photocraft-task-bar-options").num_columns(2).spacing([8.0, 4.0]).show(ui, |ui| {
        ui.label(tl!("Model"));
        // Model names are shown as the service gives them, not translated.
        let mut picked = models.iter().position(|m| m.selected).unwrap_or(usize::MAX);
        let options: Vec<(usize, &str)> = models.iter().enumerate().map(|(i, m)| (i, m.label.as_str())).collect();
        if crate::widgets::dropdown_literal(ui, "photocraft-task-bar-model", tl!("Model"), &mut picked, &options, 260.0)
            && let Some(m) = models.get(picked)
            && !m.selected
        {
            pressed = Some(Pressed::Model(m.connection.clone(), m.model.clone()));
        }
        ui.end_row();
        if !profile.qualities.is_empty() {
            // A quality the new model doesn't take goes back to Automatic.
            if !profile.qualities.contains(&bar.quality) {
                bar.quality = Quality::Auto;
            }
            ui.label(tl!("Quality"));
            let options: Vec<(Quality, &str)> = profile.qualities.iter().map(|q| (*q, q.label())).collect();
            crate::widgets::dropdown_literal(ui, "photocraft-task-bar-quality", tl!("Quality"), &mut bar.quality, &options, 260.0);
            ui.end_row();
        }
        if profile.sizes != Sizes::Unknown {
            ui.label(tl!("Resolution"));
            let mut choices = vec![SizeChoice::Auto];
            if profile.sizes == Sizes::Free {
                choices.push(SizeChoice::Canvas);
            }
            choices.extend(presets(profile.sizes).iter().map(|p| SizeChoice::Preset(*p)));
            if !choices.contains(&bar.size) {
                bar.size = SizeChoice::Auto;
            }
            let labels: Vec<(SizeChoice, String)> = choices.iter().map(|c| (*c, choice_label(*c, target))).collect();
            let options: Vec<(SizeChoice, &str)> = labels.iter().map(|(c, l)| (*c, l.as_str())).collect();
            crate::widgets::dropdown_literal(ui, "photocraft-task-bar-size", tl!("Resolution"), &mut bar.size, &options, 260.0);
            ui.end_row();
        }
    });
    pressed
}

/// The estimated cost and its explanation.
fn cost_line(bar: &Bar, model: &AiModelChoice, profile: Profile, sent: Option<[u32; 2]>, target: Target) -> (String, &'static str) {
    let quality = (!profile.qualities.is_empty()).then_some(bar.quality);
    // A model that takes no size gets the region's proportions at about 1.5 megapixels.
    let size = sent.unwrap_or_else(|| resolve(Profile { qualities: &[], sizes: Sizes::Free }, SizeChoice::Auto, target.size()).unwrap_or([1024, 1024]));
    let side = bar.references.iter().map(|r| r.size[0].max(r.size[1])).max().unwrap_or(references::SIDE);
    let order = cost::Order { size, references: bar.references.len(), reference_side: side, prompt_chars: bar.prompt.chars().count() };
    let learned = bar.learned.get(&(model.model.clone(), quality.unwrap_or(Quality::Auto).id().to_owned())).copied();
    match cost::estimate_with(&model.model, quality, order, learned) {
        Some(e) => (
            crate::i18n::fmt(tl!("Estimated cost: {cost} per image"), &[("cost", &cost::label(e))]),
            if learned.is_some() {
                tl!("An estimate from OpenAI's prices and what your last generation with this model and quality really used. What OpenAI bills can differ.")
            } else {
                tl!("An estimate from OpenAI's published prices for this model, size, quality and the pictures sent. What OpenAI bills can differ.")
            },
        ),
        None => (tl!("Estimated cost: unknown for this model").to_string(), tl!("PhotoCraft knows the prices of OpenAI's image models only.")),
    }
}

/// Whether `model` reads reference pictures besides the one it edits (the GPT Image models).
pub(crate) fn takes_references(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    gpt_image_version(&lower).is_some() || lower.contains("chatgpt-image")
}

/// The prompt's text with every attached `@tag` (and `@canvas`) in the accent colour.
fn tagged_job(ui: &Ui, text: &str, tags: &[String], accent: egui::Color32, soft: egui::Color32, wrap: f32) -> egui::text::LayoutJob {
    let font = egui::TextStyle::Body.resolve(ui.style());
    let plain = egui::TextFormat { font_id: font.clone(), color: ui.visuals().text_color(), ..Default::default() };
    let tagged = egui::TextFormat { font_id: font, color: accent, background: soft, ..Default::default() };
    let mut job = egui::text::LayoutJob::default();
    let mut last = 0;
    for (word, range) in references::tags_in(text) {
        if word != "canvas" && !tags.iter().any(|t| t.eq_ignore_ascii_case(&word)) {
            continue;
        }
        job.append(text.get(last..range.start).unwrap_or_default(), 0.0, plain.clone());
        job.append(text.get(range.clone()).unwrap_or_default(), 0.0, tagged.clone());
        last = range.end;
    }
    job.append(text.get(last..).unwrap_or_default(), 0.0, plain);
    job.wrap.max_width = wrap;
    job
}

/// The side of a reference tile, in points.
const TILE: f32 = 56.0;

/// The references as tiles (thumbnail, `@tag` under it, × to remove; a click on the tile puts the
/// tag in the description), then an Add tile while there is room.
fn references_row(ui: &mut Ui, bar: &mut Bar, takes: bool, t: &Tokens) -> Option<Pressed> {
    let mut pressed = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        let ctx = ui.ctx().clone();
        for (i, r) in bar.references.iter_mut().enumerate() {
            let tag = r.tag.clone();
            let texture = r.texture(&ctx).clone();
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 2.0;
                let (rect, resp) = ui.allocate_exact_size(egui::vec2(TILE, TILE), egui::Sense::click());
                let size = texture.size_vec2();
                let scale = TILE / size.x.max(size.y).max(1.0);
                let image = egui::Rect::from_center_size(rect.center(), size * scale);
                ui.painter().rect_filled(rect, 6.0, t.field);
                ui.painter().image(texture.id(), image, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                let border = if resp.hovered() { t.accent } else { t.field_border };
                ui.painter().rect_stroke(rect, 6.0, Stroke::new(1.0, border), egui::StrokeKind::Inside);
                if resp.on_hover_text(tl!("Put this tag in the description")).clicked() {
                    pressed = Some(Pressed::Tag(tag.clone(), None));
                }
                // The remove button over the tile's corner.
                let close = egui::Rect::from_center_size(rect.right_top() + egui::vec2(-9.0, 9.0), egui::vec2(16.0, 16.0));
                let x = ui.interact(close, ui.scope_id().with(("ai-ref-remove", i)), egui::Sense::click());
                ui.painter().circle_filled(close.center(), 8.0, if x.hovered() { t.danger } else { t.card });
                crate::icons::paint(ui, close.shrink(3.0), "x", 10.0, if x.hovered() { egui::Color32::WHITE } else { t.text });
                if x.on_hover_text(tl!("Remove this reference")).clicked() {
                    pressed = Some(Pressed::RemoveReference(i));
                }
                ui.label(RichText::new(format!("@{tag}")).small().color(t.accent));
            });
        }
        if bar.references.len() < references::MAX {
            let (rect, resp) = ui.allocate_exact_size(egui::vec2(TILE, TILE), if takes { egui::Sense::click() } else { egui::Sense::hover() });
            let color = if !takes {
                t.text_dim.gamma_multiply(0.5)
            } else if resp.hovered() {
                t.accent
            } else {
                t.text_dim
            };
            ui.painter().rect_stroke(rect, 6.0, Stroke::new(1.0, color), egui::StrokeKind::Inside);
            crate::icons::paint(ui, egui::Rect::from_center_size(rect.center() - egui::vec2(0.0, 6.0), egui::vec2(16.0, 16.0)), "plus", 16.0, color);
            ui.painter().text(rect.center() + egui::vec2(0.0, 12.0), Align2::CENTER_CENTER, tl!("Add"), egui::FontId::proportional(11.0), color);
            let tip = if takes { tl!("Attach up to three pictures (PNG, JPEG or WebP) and name them with @ in the description") } else { tl!("This model takes no reference pictures") };
            if resp.on_hover_text(tip).clicked() && takes {
                pressed = Some(Pressed::AddReference);
            }
        }
        if bar.references.is_empty() {
            ui.label(RichText::new(tl!("Optional: up to three pictures to guide the result, named with @ in the description.")).small().color(t.text_dim));
        }
    });
    pressed
}

/// Adds the pictures the file dialog answered to the bar, up to [`references::MAX`].
fn add_references(app: &mut PhotocraftApp, answer: crate::file_dialog::FileDialogAnswer) -> Result<serde_json::Value, String> {
    use crate::file_dialog::FileDialogAnswer;
    let files: Vec<(String, Result<Vec<u8>, String>)> = match answer {
        FileDialogAnswer::Contents(name, bytes) => vec![(name, Ok(bytes))],
        FileDialogAnswer::ContentsMany(many) => many.into_iter().map(|(n, b)| (n, Ok(b))).collect(),
        FileDialogAnswer::Paths(paths) => paths.into_iter().map(|p| (p.clone(), read(&p))).collect(),
        _ => return Err(tl!("Choose a picture file").into()),
    };
    let bar = &mut app.ai_select.bar;
    let mut problems = Vec::new();
    for (name, bytes) in files {
        if bar.references.len() >= references::MAX {
            problems.push(tl!("Three references at most").to_string());
            break;
        }
        let taken: Vec<String> = bar.references.iter().map(|r| r.tag.clone()).collect();
        match bytes.and_then(|b| references::load(&name, &b, &taken)) {
            Ok(r) => bar.references.push(r),
            Err(e) => problems.push(e),
        }
    }
    bar.error = problems.first().cloned();
    bar.open = true;
    Ok(json!({"references": bar.references.len()}))
}

#[cfg(not(target_arch = "wasm32"))]
fn read(path: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{path}: {e}"))
}

#[cfg(target_arch = "wasm32")]
fn read(_path: &str) -> Result<Vec<u8>, String> {
    Err(tl!("Choose a picture file").into())
}

fn apply(app: &mut PhotocraftApp, ctx: &egui::Context, pressed: Pressed, target: Target) {
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
        Pressed::AddReference => {
            let initial_dir = crate::file_dialog::last_used_dir(&app.ui.recent_files);
            let extensions = ["png", "jpg", "jpeg", "webp"].map(String::from).to_vec();
            let request = crate::file_dialog::FileDialogRequest::Open { multiple: true, initial_dir, extensions: Some(extensions) };
            if let Err(e) = app.ask_file(request, add_references)
                && e != crate::file_dialog::CANCELLED
            {
                app.ai_select.bar.error = Some(e);
            }
        }
        Pressed::RemoveReference(i) => {
            if i < bar.references.len() {
                bar.references.remove(i);
            }
        }
        Pressed::Tag(tag, at) => {
            let (text, cursor) = match at {
                Some((at, typed)) => references::complete(&bar.prompt, at, typed, &tag),
                None => {
                    let spaced = bar.prompt.is_empty() || bar.prompt.ends_with(char::is_whitespace);
                    let base = if spaced { bar.prompt.clone() } else { format!("{} ", bar.prompt) };
                    let at = base.chars().count();
                    references::complete(&base, at, 0, &tag)
                }
            };
            bar.prompt = text;
            bar.cursor_to = Some(cursor);
        }
        Pressed::Generate => {
            let current = bar.models.as_ref().and_then(|(_, m)| m.iter().find(|m| m.selected).cloned());
            let profile = current.as_ref().map_or(Profile { qualities: &[], sizes: Sizes::Unknown }, |m| profile(&m.model));
            let takes = current.as_ref().is_some_and(|m| takes_references(&m.model));
            let options = FillOptions {
                quality: profile.qualities.contains(&bar.quality).then(|| bar.quality.id().to_string()),
                size: resolve(profile, bar.size, target.size()),
                references: if takes { bar.references.iter().map(|r| (r.tag.clone(), r.png.clone())).collect() } else { Vec::new() },
            };
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
        assert_eq!(profile("gpt-image-2"), Profile { qualities: BASE_QUALITIES, sizes: Sizes::Free });
        assert_eq!(profile("gpt-image-2-2026-04-21"), Profile { qualities: BASE_QUALITIES, sizes: Sizes::Free });
        assert_eq!(profile("gpt-image-2.5-flare-2026-09-08"), Profile { qualities: ALL_QUALITIES, sizes: Sizes::Free });
        assert_eq!(profile("gpt-image-2.5-sunburst"), Profile { qualities: ALL_QUALITIES, sizes: Sizes::Free });
        assert_eq!(profile("gpt-image-2.6-flare"), Profile { qualities: ALL_QUALITIES, sizes: Sizes::Free });
        assert_eq!(profile("gpt-image-1.5"), Profile { qualities: BASE_QUALITIES, sizes: Sizes::Fixed(GPT_IMAGE_1_SIZES) });
        assert_eq!(profile("gpt-image-1-mini"), Profile { qualities: BASE_QUALITIES, sizes: Sizes::Fixed(GPT_IMAGE_1_SIZES) });
        assert_eq!(profile("chatgpt-image-latest"), Profile { qualities: BASE_QUALITIES, sizes: Sizes::Fixed(GPT_IMAGE_1_SIZES) });
        assert_eq!(profile("dall-e-2"), Profile { qualities: &[], sizes: Sizes::Fixed(DALL_E_2_SIZES) });
        assert_eq!(profile("flux-fill"), Profile { qualities: &[], sizes: Sizes::Unknown });
        assert_eq!(Quality::XHigh.id(), "xhigh");
        assert_eq!(Quality::Max.id(), "max");
    }

    fn multiple_of_16(s: [u32; 2]) -> bool {
        s[0].is_multiple_of(16) && s[1].is_multiple_of(16)
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
