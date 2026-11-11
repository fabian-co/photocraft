//! Edit › Generative Fill: describe what should be in the selection and an AI image model paints
//! it, as a new layer masked by the selection (non-destructive, one history step).
//!
//! The model runs in the desktop shell (the Craft AI layer), never here: `edit.generativeFill`
//! is the menu item (the shell opens its prompt; headless it says what it needs), and two
//! commands do the document side: `edit.aiFillSource` hands out the picture and the selection as
//! the model reads them, and `layer.fromAiImage` turns the picture that comes back into a layer.

use base64::Engine as _;
use photocraft_doc::{ColorMode, Layer};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::commands::{mask_at_depth, selection_mask};
use crate::smartselect_cmds::with_doc_sampler;
use crate::{EngineError, Result, Session};

/// Longest side of the picture sent to an image model (OpenAI's edits accept up to 1536 px).
const MAX_SIDE: u32 = 1536;
/// Image models work on blocks of 16 pixels: every side sent is a multiple of this.
pub const SIDE_STEP: u32 = 16;
/// The largest side that can be asked for.
const MAX_ASKED: u32 = 4096;

/// `v` rounded to the nearest multiple of [`SIDE_STEP`], at least one step and at most
/// [`MAX_ASKED`].
pub fn round_side(v: f64) -> u32 {
    let steps = if v.is_finite() { (v / f64::from(SIDE_STEP)).round() } else { 1.0 };
    (steps.clamp(1.0, f64::from(MAX_ASKED / SIDE_STEP)) as u32) * SIDE_STEP
}

fn bad(cmd: &str, msg: &str) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

/// Generative Fill needs an RGB document. Without a selection it works on the whole canvas.
fn enabled(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.pixel_format().mode != ColorMode::Rgb {
        return Err("Generative Fill works on RGB documents".into());
    }
    Ok(())
}

/// Context kept around a selection: this share of its longest side on every side…
const CONTEXT_SHARE: f64 = 0.6;
/// …and at least this many pixels.
const CONTEXT_MIN: f64 = 96.0;
/// The region sent is never longer than this many times its height (or the other way round),
/// when the canvas leaves room: image models take 3:1 at most.
const REGION_RATIO: f64 = 2.5;

/// The part of the canvas `bounds` sent to an image model for a selection whose bounds are
/// `selection`: the selection with some of the picture around it, so the model sees the area at a
/// useful size and still has context; the whole canvas without a selection. Always inside
/// `bounds` and never empty.
pub fn fill_region(bounds: Rect, selection: Option<Rect>) -> Rect {
    fill_region_shaped(bounds, selection, None)
}

/// [`fill_region`] widened (or heightened) to `aspect` (width / height) where the canvas leaves
/// room, so a model that only makes a few sizes gets the region at its proportions instead of
/// stretched. The whole canvas is never reshaped.
pub fn fill_region_shaped(bounds: Rect, selection: Option<Rect>, aspect: Option<f64>) -> Rect {
    let Some(sel) = selection.map(|s| s.intersect(&bounds)).filter(|s| !s.is_empty()) else { return bounds };
    let longest = f64::from(sel.width().max(sel.height()));
    let pad = (longest * CONTEXT_SHARE).max(CONTEXT_MIN);
    let (mut x0, mut y0) = (f64::from(sel.x0) - pad, f64::from(sel.y0) - pad);
    let (mut x1, mut y1) = (f64::from(sel.x1) + pad, f64::from(sel.y1) + pad);
    // Not too long and thin: grow the short side around its middle.
    let (w, h) = (x1 - x0, y1 - y0);
    if w > h * REGION_RATIO {
        let grow = (w / REGION_RATIO - h) / 2.0;
        (y0, y1) = (y0 - grow, y1 + grow);
    } else if h > w * REGION_RATIO {
        let grow = (h / REGION_RATIO - w) / 2.0;
        (x0, x1) = (x0 - grow, x1 + grow);
    }
    if let Some(aspect) = aspect.filter(|a| a.is_finite() && *a > 0.0) {
        let (w, h) = (x1 - x0, y1 - y0);
        if w / h < aspect {
            let grow = (h * aspect - w) / 2.0;
            (x0, x1) = (x0 - grow, x1 + grow);
        } else {
            let grow = (w / aspect - h) / 2.0;
            (y0, y1) = (y0 - grow, y1 + grow);
        }
    }
    // Slide back inside the canvas before clipping, so the context is kept where there is room.
    let shift = |lo: f64, hi: f64, min: f64, max: f64| {
        if lo < min {
            (min, (hi + (min - lo)).min(max))
        } else if hi > max {
            ((lo - (hi - max)).max(min), max)
        } else {
            (lo, hi)
        }
    };
    let (bx0, by0, bx1, by1) = (f64::from(bounds.x0), f64::from(bounds.y0), f64::from(bounds.x1), f64::from(bounds.y1));
    (x0, x1) = shift(x0, x1, bx0, bx1);
    (y0, y1) = shift(y0, y1, by0, by1);
    let region = Rect::new(x0.floor() as i32, y0.floor() as i32, x1.ceil() as i32, y1.ceil() as i32).intersect(&bounds);
    if region.is_empty() { bounds } else { region }
}

fn generative_fill(s: &mut Session, _: &Value) -> Result<Value> {
    enabled(s).map_err(EngineError::Other)?;
    Err(EngineError::Other("Generative Fill needs an AI image model: choose one in Edit › Preferences › Integrations in the desktop app".into()))
}

fn encode_png(cmd: &str, w: u32, h: u32, rgba: &[u8]) -> Result<String> {
    let mut image =
        photocraft_codecs::Image::new(w, h, photocraft_codecs::ChannelLayout::Rgba, photocraft_codecs::SampleType::U8).map_err(|e| bad(cmd, &e.to_string()))?;
    image.data_mut().copy_from_slice(rgba);
    let png = photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &photocraft_codecs::EncodeOptions::default())
        .map_err(|e| bad(cmd, &e.to_string()))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(png))
}

/// What part of the picture holds, as the model is told: nothing (transparent), one plain colour
/// (a new document's background), or content to keep in mind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Contents {
    Nothing,
    Plain,
    Content,
}

impl Contents {
    pub fn id(self) -> &'static str {
        match self {
            Self::Nothing => "nothing",
            Self::Plain => "plain",
            Self::Content => "content",
        }
    }
}

/// Opaque coverage and colour spread of the pixels sampled for [`Contents`].
#[derive(Default)]
struct Look {
    pixels: f64,
    opaque: f64,
    sum: [f64; 3],
    squares: [f64; 3],
}

/// Fewer opaque pixels than this share are "nothing"; a colour spread (standard deviation, 0..1)
/// under `PLAIN_SPREAD` is one plain colour.
const OPAQUE_SHARE: f64 = 0.02;
const PLAIN_SPREAD: f64 = 0.03;

impl Look {
    fn add(&mut self, px: [u8; 4]) {
        self.pixels += 1.0;
        if px[3] < 13 {
            return;
        }
        self.opaque += 1.0;
        for ((value, sum), square) in px.iter().zip(self.sum.iter_mut()).zip(self.squares.iter_mut()) {
            let v = f64::from(*value) / 255.0;
            *sum += v;
            *square += v * v;
        }
    }

    fn kind(&self) -> Contents {
        if self.pixels <= 0.0 || self.opaque / self.pixels < OPAQUE_SHARE {
            return Contents::Nothing;
        }
        let variance: f64 = (0..3).map(|c| (self.squares[c] / self.opaque - (self.sum[c] / self.opaque).powi(2)).max(0.0)).sum::<f64>() / 3.0;
        if variance.sqrt() < PLAIN_SPREAD { Contents::Plain } else { Contents::Content }
    }
}

/// The picture (the composite, or the active layer) and the selection as an image model reads
/// them: two RGBA PNGs of the same size, of the region [`fill_region`] picks (with `crop`, the
/// default: the selection and the picture around it; else, or without a selection, the whole
/// canvas). That size is `width` × `height` when given, else the region scaled to at most
/// `maxSide` on the longest side; either way both sides are rounded to a multiple of 16
/// ([`round_side`]), stretching the picture a little if need be. The mask is transparent where
/// the model should paint (the selection, or everything without one) and opaque elsewhere.
/// `selection` and `canvas` say what the painted area and the whole region hold ([`Contents`]),
/// so the prompt can ask the model to keep what is there in mind, or to create freely on a blank
/// canvas. `region` is `[x, y, width, height]` in canvas pixels, for `layer.fromAiImage`.
fn fill_source(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "edit.aiFillSource";
    enabled(s).map_err(|e| bad(CMD, &e))?;
    let all_layers = p.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(true);
    let crop = p.get("crop").and_then(Value::as_bool).unwrap_or(true);
    let max_side = p.get("maxSide").and_then(Value::as_u64).map_or(MAX_SIDE, |v| v.clamp(64, 4096) as u32);
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let selection = d.doc.selection.as_ref();
    let canvas_bounds = d.doc.bounds();
    if canvas_bounds.is_empty() {
        return Err(bad(CMD, "the document is empty"));
    }
    let asked = |key: &str| p.get(key).and_then(Value::as_f64);
    // A size asked for shapes the region, so the picture is not stretched to it.
    let aspect = match (asked("width"), asked("height")) {
        (Some(w), Some(h)) if w > 0.0 && h > 0.0 => Some(w / h),
        _ => None,
    };
    let bounds = if crop { fill_region_shaped(canvas_bounds, selection.map(|s| s.content_bounds()), aspect) } else { canvas_bounds };
    let (sw, sh) = (bounds.width(), bounds.height());
    if sw == 0 || sh == 0 {
        return Err(bad(CMD, "the document is empty"));
    }
    let (w, h) = match (asked("width"), asked("height")) {
        (Some(w), Some(h)) => (round_side(w), round_side(h)),
        (None, None) => {
            let scale = (f64::from(max_side) / f64::from(sw.max(sh))).min(1.0);
            (round_side(f64::from(sw) * scale), round_side(f64::from(sh) * scale))
        }
        _ => return Err(bad(CMD, "give both width and height, or neither")),
    };
    let (scale_x, scale_y) = (f64::from(w) / f64::from(sw), f64::from(h) / f64::from(sh));
    let len = (w as usize).saturating_mul(h as usize).saturating_mul(4);
    let mut picture = vec![0u8; len];
    let mut mask = vec![0u8; len];
    let source_x = |x: u32| (((f64::from(x) + 0.5) / scale_x) as i32).min(sw as i32 - 1);
    let source_y = |y: u32| (((f64::from(y) + 0.5) / scale_y) as i32).min(sh as i32 - 1);
    with_doc_sampler(&d.doc, d.active_layer, all_layers, |smp, _| {
        for y in 0..h {
            let sy = bounds.y0 + source_y(y);
            let row = smp.rgba(Rect::new(bounds.x0, sy, bounds.x1, sy + 1));
            for x in 0..w {
                let px = row.get(source_x(x) as usize).copied().unwrap_or([0.0; 4]);
                let at = (y as usize * w as usize + x as usize) * 4;
                for c in 0..4 {
                    picture[at + c] = (px[c].clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
    });
    let mut any = false;
    let (mut inside, mut canvas) = (Look::default(), Look::default());
    for y in 0..h {
        for x in 0..w {
            let covered = selection.map_or(1.0, |sel| sel.sample_channel(bounds.x0 + source_x(x), bounds.y0 + source_y(y), 0).clamp(0.0, 1.0));
            any |= covered > 0.0;
            let at = (y as usize * w as usize + x as usize) * 4;
            let px = [picture[at], picture[at + 1], picture[at + 2], picture[at + 3]];
            canvas.add(px);
            if covered >= 0.5 {
                inside.add(px);
            }
            mask[at..at + 3].fill(255);
            mask[at + 3] = ((1.0 - covered) * 255.0).round() as u8;
        }
    }
    if !any {
        return Err(bad(CMD, "the selection is empty"));
    }
    Ok(json!({
        "png": encode_png(CMD, w, h, &picture)?,
        "mask": encode_png(CMD, w, h, &mask)?,
        "width": w,
        "height": h,
        "document": d.doc.id.0,
        "selection": inside.kind().id(),
        "canvas": canvas.kind().id(),
        "region": [bounds.x0, bounds.y0, sw, sh],
    }))
}

/// Fewest pixels outside the selection that [`match_colors`] fits a correction to.
const MATCH_MIN: usize = 500;
/// Most samples it reads (evenly spaced), whatever the region's size.
const MATCH_SAMPLES: usize = 200_000;

/// Per channel, the gain and offset that bring `data` (RGBA, the model's picture over the
/// region) closest to `original` where `coverage` (the selection) is 0, applied to all of
/// `data`. Nothing changes without enough such pixels, and the gain stays within 0.6..1.6.
fn match_colors(data: &mut [f32], original: &[[f32; 4]], coverage: &[f32]) {
    let n = coverage.len().min(original.len()).min(data.len() / 4);
    let step = (n / MATCH_SAMPLES).max(1);
    let outside: Vec<usize> = (0..n)
        .step_by(step)
        .filter(|&i| coverage.get(i).is_some_and(|c| *c < 0.01) && original.get(i).is_some_and(|o| o[3] > 0.5) && data.get(i * 4 + 3).is_some_and(|a| *a > 0.5))
        .collect();
    if outside.len() < MATCH_MIN {
        return;
    }
    let count = outside.len() as f64;
    let mut fit = [(1.0f32, 0.0f32); 3];
    for (c, slot) in fit.iter_mut().enumerate() {
        let (mut sg, mut so, mut sgg, mut sgo) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for &i in &outside {
            let g = f64::from(data[i * 4 + c]);
            let o = f64::from(original[i][c]);
            sg += g;
            so += o;
            sgg += g * g;
            sgo += g * o;
        }
        let (mg, mo) = (sg / count, so / count);
        let var = sgg / count - mg * mg;
        let gain = if var > 1e-5 { ((sgo / count - mg * mo) / var).clamp(0.6, 1.6) } else { 1.0 };
        *slot = (gain as f32, (mo - gain * mg) as f32);
    }
    for px in data.chunks_mut(4) {
        for (v, (gain, offset)) in px.iter_mut().zip(fit) {
            *v = (*v * gain + offset).clamp(0.0, 1.0);
        }
    }
}

/// A colour difference (0..1, the largest of R, G, B) below this is noise; above `CHANGE_HIGH`
/// it is surely a change.
const CHANGE_LOW: f32 = 0.10;
const CHANGE_HIGH: f32 = 0.20;
/// The share of the region's longest side the change mask is smoothed and then grown by.
const CHANGE_SMOOTH: f32 = 0.004;
const CHANGE_GROW: f32 = 0.012;

/// Where the model's picture `data` (RGBA over a `w` × `h` region) differs from `original`:
/// 1 where something changed, 0 where it is the same picture, with a soft, slightly grown edge
/// so a change is kept whole.
fn change_mask(data: &[f32], original: &[[f32; 4]], w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let diff: Vec<f32> = (0..n)
        .map(|i| {
            let (Some(g), Some(o)) = (data.get(i * 4..i * 4 + 4), original.get(i)) else { return 0.0 };
            (0..3).map(|c| (g[c] - o[c]).abs()).fold(0.0, f32::max)
        })
        .collect();
    let longest = w.max(h) as f32;
    let smooth = photocraft_algo::selection::feather(&diff, w, h, (longest * CHANGE_SMOOTH).max(1.5));
    let changed: Vec<f32> = smooth
        .iter()
        .map(|d| {
            let t = ((d - CHANGE_LOW) / (CHANGE_HIGH - CHANGE_LOW)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        })
        .collect();
    // Grow and soften: doubling a feathered mask pushes its half-way line outwards.
    let grown = photocraft_algo::selection::feather(&changed, w, h, (longest * CHANGE_GROW).max(3.0));
    grown.iter().map(|v| (v * 2.0).clamp(0.0, 1.0)).collect()
}

/// The default soft edge of a Generative Fill layer's mask: this share of the region's longest
/// side, and at least `FEATHER_MIN` pixels.
const FEATHER_SHARE: f32 = 0.006;
const FEATHER_MIN: f32 = 2.0;

/// Feathers `mask` (a selection-shaped layer mask) by `radius` inside `region`, where the new
/// layer has pixels; outside it the mask stays as it is.
fn soften(mask: &mut photocraft_raster::Surface, region: Rect, radius: f32) {
    let channels = mask.channels().max(1);
    let (w, h) = (region.width() as usize, region.height() as usize);
    if w == 0 || h == 0 {
        return;
    }
    let raw = mask.read_region(region);
    let coverage: Vec<f32> = raw.chunks(channels).map(|c| c.first().copied().unwrap_or(0.0)).collect();
    if coverage.len() != w * h {
        return;
    }
    let soft = photocraft_algo::selection::feather(&coverage, w, h, radius);
    let data: Vec<f32> = soft.iter().flat_map(|v| std::iter::repeat_n(v.clamp(0.0, 1.0), channels)).collect();
    mask.write_region(region, &data);
}

/// The picture an image model returned, stretched over `region` (`[x, y, width, height]`, as
/// `edit.aiFillSource` gave it; the whole canvas without it) as a new layer above the active one,
/// masked by the selection when there is one: one undoable step.
fn from_ai_image(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "layer.fromAiImage";
    let encoded = p.get("png").and_then(Value::as_str).ok_or_else(|| bad(CMD, "png (base64) is required"))?;
    let bytes = base64::engine::general_purpose::STANDARD.decode(encoded).map_err(|_| bad(CMD, "png is not valid base64"))?;
    let image = photocraft_codecs::decode(&bytes).map_err(|e| bad(CMD, &format!("the picture could not be read: {e}")))?;
    let (iw, ih) = (image.width() as usize, image.height() as usize);
    let rgba = image.to_rgba8();
    if iw == 0 || ih == 0 || rgba.len() < iw * ih * 4 {
        return Err(bad(CMD, "the picture is empty"));
    }
    let name = p.get("name").and_then(Value::as_str).unwrap_or("Generative Fill").to_string();
    let d = s.active().ok_or(EngineError::NoDocument)?;
    if let Some(doc) = p.get("document").and_then(Value::as_u64)
        && doc != d.doc.id.0
    {
        return Err(bad(CMD, "the picture was made for another document"));
    }
    if d.doc.pixel_format().mode != ColorMode::Rgb {
        return Err(bad(CMD, "Generative Fill works on RGB documents"));
    }
    let canvas = d.doc.bounds();
    let bounds = match p.get("region") {
        None | Some(Value::Null) => canvas,
        Some(v) => {
            let n: Vec<i64> = v.as_array().map(|a| a.iter().filter_map(Value::as_i64).collect()).unwrap_or_default();
            let [x, y, w, h] = n[..] else { return Err(bad(CMD, "region is [x, y, width, height]")) };
            let fits = |v: i64| i32::try_from(v).ok();
            let (Some(x), Some(y), Some(w), Some(h)) = (fits(x), fits(y), fits(w), fits(h)) else { return Err(bad(CMD, "region is out of range")) };
            if w <= 0 || h <= 0 {
                return Err(bad(CMD, "region is empty"));
            }
            let r = Rect::new(x, y, x.saturating_add(w), y.saturating_add(h));
            if !canvas.contains_rect(&r) {
                return Err(bad(CMD, "region is outside the canvas"));
            }
            r
        }
    };
    let (dw, dh) = (bounds.width() as usize, bounds.height() as usize);
    // Bilinear stretch over the canvas, straight (not premultiplied) colour.
    let at = |x: usize, y: usize, c: usize| f32::from(rgba[(y * iw + x) * 4 + c]) / 255.0;
    let mut data = Vec::with_capacity(dw.saturating_mul(dh).saturating_mul(4));
    for y in 0..dh {
        let fy = ((y as f64 + 0.5) * ih as f64 / dh as f64 - 0.5).clamp(0.0, (ih - 1) as f64);
        let (y0, ty) = (fy.floor() as usize, (fy - fy.floor()) as f32);
        let y1 = (y0 + 1).min(ih - 1);
        for x in 0..dw {
            let fx = ((x as f64 + 0.5) * iw as f64 / dw as f64 - 0.5).clamp(0.0, (iw - 1) as f64);
            let (x0, tx) = (fx.floor() as usize, (fx - fx.floor()) as f32);
            let x1 = (x0 + 1).min(iw - 1);
            for c in 0..4 {
                let top = at(x0, y0, c) * (1.0 - tx) + at(x1, y0, c) * tx;
                let bottom = at(x0, y1, c) * (1.0 - tx) + at(x1, y1, c) * tx;
                data.push(top * (1.0 - ty) + bottom * ty);
            }
        }
    }
    let keep_changes = match p.get("keep").and_then(Value::as_str) {
        None | Some("all") => false,
        Some("changes") => true,
        Some(_) => return Err(bad(CMD, "keep is \"all\" or \"changes\"")),
    };
    let match_color = p.get("matchColor").and_then(Value::as_bool).unwrap_or(true);
    // Image models redraw the whole picture and drift in colour: bring the new picture's colours
    // back to the original's where they should agree (outside the selection, inside the region).
    let mut changes = None;
    if let Some(selection) = d.doc.selection.as_ref() {
        if match_color {
            let original = with_doc_sampler(&d.doc, None, true, |smp, _| smp.rgba(bounds));
            let raw = selection.read_region(bounds);
            let channels = selection.channels().max(1);
            let coverage: Vec<f32> = raw.chunks(channels).map(|c| c.first().copied().unwrap_or(0.0)).collect();
            match_colors(&mut data, &original, &coverage);
        }
    } else if keep_changes {
        // An instruction for the whole picture ("make the chair hay"): keep only what the model
        // changed, so the rest of the photo stays as it was, at its own resolution.
        let original = with_doc_sampler(&d.doc, None, true, |smp, _| smp.rgba(bounds));
        if match_color {
            match_colors(&mut data, &original, &vec![0.0; original.len()]);
        }
        changes = Some(change_mask(&data, &original, bounds.width() as usize, bounds.height() as usize));
    }
    // A soft edge so the new content meets the picture without a seam.
    let feather = match p.get("feather").and_then(Value::as_f64) {
        Some(f) if f.is_finite() => f.clamp(0.0, 250.0) as f32,
        Some(_) => return Err(bad(CMD, "feather is a number of pixels")),
        None => (bounds.width().max(bounds.height()) as f32 * FEATHER_SHARE).max(FEATHER_MIN),
    };
    let label = name.clone();
    let id = s.edit(&label, move |doc, active| {
        let mut layer = Layer::raster(name, doc.pixel_format());
        let surface = layer.surface_mut().ok_or_else(|| bad(CMD, "the new layer has no pixels"))?;
        surface.write_region(bounds, &data);
        layer.mask = selection_mask(doc)?;
        if let Some(mask) = layer.mask.as_mut()
            && feather > 0.0
        {
            soften(&mut mask.surface, bounds, feather);
        }
        if let Some(changed) = &changes
            && let Some(surface) = photocraft_algo::selection::combine_band(None, changed, bounds, photocraft_algo::selection::SelectionMode::Replace)
        {
            layer.mask = Some(mask_at_depth(photocraft_doc::LayerMask { surface, ..photocraft_doc::LayerMask::reveal_all() }, doc.depth)?);
        }
        let id = doc.insert_above(*active, layer);
        *active = Some(id);
        Ok(id)
    })?;
    Ok(json!({"layer": id.0}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "edit.generativeFill",
            label: "Generative Fill…",
            menu: &["Edit"],
            shortcut: None,
            params: r##"{} → the desktop app asks for a description and paints the selection with the chosen AI image model (Edit › Preferences › Integrations); headless it reports that it needs one"##,
            enabled,
            run: generative_fill,
            journal: false,
        },
        CommandSpec {
            id: "edit.aiFillSource",
            label: "Generative Fill Source",
            menu: &[],
            shortcut: None,
            params: r##"{"sampleAllLayers":bool=true,"maxSide":64..4096=1536,"width":px?,"height":px?,"crop":bool=true} → {png,mask,width,height,document,selection,canvas,region} (RGBA PNGs, base64, both sides multiples of 16, of the selection with context around it, or the whole canvas; the mask is transparent where the selection is, everywhere without one; selection and canvas say what they hold: "nothing", "plain" or "content"; region is [x,y,w,h] in canvas pixels)"##,
            enabled,
            run: fill_source,
            journal: false,
        },
        CommandSpec {
            id: "layer.fromAiImage",
            label: "Generative Fill",
            menu: &[],
            shortcut: None,
            params: r##"{"png":base64,"document":id?,"name":str="Generative Fill","region":[x,y,w,h]?,"feather":px?,"matchColor":bool=true,"keep":"all|changes"="all"} → {layer} (stretched over the region, else the canvas, above the active layer, masked by the selection when there is one, its edge feathered by 0.6 % of the region's longest side, at least 2 px, unless feather says otherwise; without a selection, keep "changes" masks the layer to where the picture differs from the original)"##,
            enabled: |s| s.active().map(|_| ()).ok_or_else(|| "no document open".into()),
            run: from_ai_image,
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests;
