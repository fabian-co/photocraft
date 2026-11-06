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
use crate::commands::selection_mask;
use crate::smartselect_cmds::with_doc_sampler;
use crate::{EngineError, Result, Session};

/// Longest side of the picture sent to an image model (OpenAI's edits accept up to 1536 px).
const MAX_SIDE: u32 = 1536;

fn bad(cmd: &str, msg: &str) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

/// Generative Fill needs an RGB document with a selection, as in Photoshop.
fn enabled(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.pixel_format().mode != ColorMode::Rgb {
        return Err("Generative Fill works on RGB documents".into());
    }
    if d.doc.selection.is_none() {
        return Err("make a selection first".into());
    }
    Ok(())
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

/// The picture (the composite, or the active layer) and the selection as an image model reads
/// them: two RGBA PNGs of the same size, at most `maxSide` on the longest side. The mask is
/// transparent where the model should paint (the selection) and opaque elsewhere.
fn fill_source(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "edit.aiFillSource";
    enabled(s).map_err(|e| bad(CMD, &e))?;
    let all_layers = p.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(true);
    let max_side = p.get("maxSide").and_then(Value::as_u64).map_or(MAX_SIDE, |v| v.clamp(64, 4096) as u32);
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let selection = d.doc.selection.as_ref().ok_or_else(|| bad(CMD, "make a selection first"))?;
    let bounds = d.doc.bounds();
    let (sw, sh) = (bounds.width(), bounds.height());
    if sw == 0 || sh == 0 {
        return Err(bad(CMD, "the document is empty"));
    }
    let scale = (f64::from(max_side) / f64::from(sw.max(sh))).min(1.0);
    let w = ((f64::from(sw) * scale).round() as u32).max(1);
    let h = ((f64::from(sh) * scale).round() as u32).max(1);
    let len = (w as usize).saturating_mul(h as usize).saturating_mul(4);
    let mut picture = vec![0u8; len];
    let mut mask = vec![0u8; len];
    let source_x = |x: u32| (((f64::from(x) + 0.5) / scale) as i32).min(sw as i32 - 1);
    let source_y = |y: u32| (((f64::from(y) + 0.5) / scale) as i32).min(sh as i32 - 1);
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
    for y in 0..h {
        for x in 0..w {
            let covered = selection.sample_channel(bounds.x0 + source_x(x), bounds.y0 + source_y(y), 0).clamp(0.0, 1.0);
            any |= covered > 0.0;
            let at = (y as usize * w as usize + x as usize) * 4;
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
    }))
}

/// The picture an image model returned, stretched over the canvas as a new layer above the
/// active one, masked by the selection: one undoable step.
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
    let bounds = d.doc.bounds();
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
    let label = name.clone();
    let id = s.edit(&label, move |doc, active| {
        let mut layer = Layer::raster(name, doc.pixel_format());
        let surface = layer.surface_mut().ok_or_else(|| bad(CMD, "the new layer has no pixels"))?;
        surface.write_region(bounds, &data);
        layer.mask = selection_mask(doc)?;
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
            params: r##"{"sampleAllLayers":bool=true,"maxSide":64..4096=1536} → {png,mask,width,height,document} (RGBA PNGs, base64; the mask is transparent where the selection is)"##,
            enabled,
            run: fill_source,
            journal: false,
        },
        CommandSpec {
            id: "layer.fromAiImage",
            label: "Generative Fill",
            menu: &[],
            shortcut: None,
            params: r##"{"png":base64,"document":id?,"name":str="Generative Fill"} → {layer} (stretched over the canvas, above the active layer, masked by the selection)"##,
            enabled: |s| s.active().map(|_| ()).ok_or_else(|| "no document open".into()),
            run: from_ai_image,
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests;
