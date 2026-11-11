//! Generative Fill's reference images and the words the model is sent.
//!
//! Up to [`MAX`] pictures can go with the description. Each gets a tag (`@dog`, from its file
//! name) that the description can name, so "put the hat from @hat on @dog" says which picture is
//! which: the model is told that the canvas is image 1 and the references images 2, 3 and 4, and
//! every tag in the description is followed by its image's number.
//!
//! The words also say what the selection holds (`edit.aiFillSource`'s `selection` and `canvas`):
//! content there is context to keep in mind, an empty area gets new content that blends in, and a
//! blank canvas (a new document) leaves the model free to create.

use egui::{ColorImage, TextureHandle};

/// The most reference images one fill takes.
pub(crate) const MAX: usize = 3;
/// A reference's longest side once loaded: plenty for a model to read, and cheaper to send.
pub(crate) const SIDE: u32 = 1024;
/// The largest file read as a reference.
const MAX_FILE: usize = 64 << 20;
/// The thumbnail's longest side, in pixels.
const THUMB: u32 = 64;
/// The longest tag.
const MAX_TAG: usize = 24;

/// One picture to send with the description.
pub(crate) struct Reference {
    /// Without the `@`.
    pub tag: String,
    /// PNG, at most [`SIDE`] on its longest side.
    pub png: Vec<u8>,
    pub size: [u32; 2],
    thumb: ColorImage,
    texture: Option<TextureHandle>,
}

impl Reference {
    /// The thumbnail, made into a texture the first time it is drawn.
    pub(crate) fn texture(&mut self, ctx: &egui::Context) -> &TextureHandle {
        let thumb = &self.thumb;
        let tag = &self.tag;
        self.texture.get_or_insert_with(|| ctx.load_texture(format!("ai-fill-reference-{tag}"), thumb.clone(), egui::TextureOptions::LINEAR))
    }
}

/// A tag made from `name` (a file name): its stem in lower case, letters, digits, `_` and `-`
/// only, unlike every tag in `taken`. `img1`, `img2`… when nothing is left.
pub(crate) fn tag_for(name: &str, taken: &[String]) -> String {
    let file = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let stem = file.rsplit_once('.').map_or(file, |(s, _)| s);
    let mut base: String =
        stem.chars().map(|c| if c.is_whitespace() { '_' } else { c }).filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-').flat_map(char::to_lowercase).take(MAX_TAG).collect();
    if base.is_empty() || base.chars().all(|c| c == '_' || c == '-') {
        base = "img".into();
    }
    let free = |t: &str| !taken.iter().any(|o| o.eq_ignore_ascii_case(t));
    if base != "img" && free(&base) {
        return base;
    }
    (1..).map(|n| format!("{base}{n}")).find(|t| free(t)).unwrap_or(base)
}

/// Shrinks straight RGBA8 `w` × `h` so its longest side is at most `side`, averaging boxes.
fn shrink(rgba: &[u8], w: u32, h: u32, side: u32) -> (Vec<u8>, u32, u32) {
    let longest = w.max(h).max(1);
    if longest <= side {
        return (rgba.to_vec(), w, h);
    }
    let scale = f64::from(side) / f64::from(longest);
    let nw = ((f64::from(w) * scale).round() as u32).max(1);
    let nh = ((f64::from(h) * scale).round() as u32).max(1);
    let mut out = Vec::with_capacity(nw as usize * nh as usize * 4);
    for y in 0..nh {
        let y0 = (u64::from(y) * u64::from(h) / u64::from(nh)) as u32;
        let y1 = ((u64::from(y + 1) * u64::from(h) / u64::from(nh)) as u32).clamp(y0 + 1, h);
        for x in 0..nw {
            let x0 = (u64::from(x) * u64::from(w) / u64::from(nw)) as u32;
            let x1 = ((u64::from(x + 1) * u64::from(w) / u64::from(nw)) as u32).clamp(x0 + 1, w);
            let mut sum = [0u64; 4];
            let mut n = 0u64;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let at = (sy as usize * w as usize + sx as usize) * 4;
                    if let Some(px) = rgba.get(at..at + 4) {
                        for c in 0..4 {
                            sum[c] += u64::from(px[c]);
                        }
                        n += 1;
                    }
                }
            }
            let n = n.max(1);
            out.extend(sum.map(|v| (v / n) as u8));
        }
    }
    (out, nw, nh)
}

/// Reads a PNG, JPEG, WebP (or any picture the codecs read) called `name` into a reference tagged
/// unlike `taken`.
pub(crate) fn load(name: &str, bytes: &[u8], taken: &[String]) -> Result<Reference, String> {
    if bytes.len() > MAX_FILE {
        return Err(tl!("That picture is too large to use as a reference").into());
    }
    let image = photocraft_codecs::decode(bytes).map_err(|_| tl!("That file is not a picture PhotoCraft can read").to_string())?;
    let (w, h) = (image.width(), image.height());
    let rgba = image.to_rgba8();
    if w == 0 || h == 0 || rgba.len() < (w as usize).saturating_mul(h as usize).saturating_mul(4) {
        return Err(tl!("That file is not a picture PhotoCraft can read").into());
    }
    let (sent, sw, sh) = shrink(&rgba, w, h, SIDE);
    let mut out = photocraft_codecs::Image::new(sw, sh, photocraft_codecs::ChannelLayout::Rgba, photocraft_codecs::SampleType::U8).map_err(|e| e.to_string())?;
    out.data_mut().copy_from_slice(&sent);
    let png = photocraft_codecs::encode(&out, photocraft_codecs::Format::Png, &photocraft_codecs::EncodeOptions::default()).map_err(|e| e.to_string())?;
    let (thumb, tw, th) = shrink(&sent, sw, sh, THUMB);
    let thumb = ColorImage::from_rgba_unmultiplied([tw as usize, th as usize], &thumb);
    Ok(Reference { tag: tag_for(name, taken), png, size: [sw, sh], thumb, texture: None })
}

/// The partial tag being typed just before char index `cursor` of `text`: the letters after an
/// `@` that starts a word. `None` when the cursor is not in a tag.
pub(crate) fn typing_tag(text: &str, cursor: usize) -> Option<(usize, String)> {
    let before: Vec<char> = text.chars().take(cursor).collect();
    let at = before.iter().rposition(|c| *c == '@')?;
    let word: String = before.get(at + 1..)?.iter().collect();
    if word.chars().any(|c| !(c.is_alphanumeric() || c == '_' || c == '-')) {
        return None;
    }
    if at > 0 && before.get(at - 1).is_some_and(|c| c.is_alphanumeric()) {
        return None; // an e-mail address, not a tag
    }
    Some((at, word))
}

/// `text` with the partial tag that starts at char `at` and runs `typed` chars replaced by `@tag `;
/// the new cursor position, in chars.
pub(crate) fn complete(text: &str, at: usize, typed: usize, tag: &str) -> (String, usize) {
    let chars: Vec<char> = text.chars().collect();
    let end = (at + 1 + typed).min(chars.len());
    let mut out: String = chars.get(..at.min(chars.len())).unwrap_or_default().iter().collect();
    out.push('@');
    out.push_str(tag);
    let mut rest: String = chars.get(end..).unwrap_or_default().iter().collect();
    if !rest.starts_with(' ') {
        rest.insert(0, ' ');
    }
    let cursor = out.chars().count() + 1;
    out.push_str(&rest);
    (out, cursor)
}

/// The tags `text` names that are not attached, in order of appearance.
pub(crate) fn unknown_tags(text: &str, tags: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for (word, _) in tags_in(text) {
        if !tags.iter().any(|t| t.eq_ignore_ascii_case(&word)) && !out.contains(&word) && word != "canvas" {
            out.push(word);
        }
    }
    out
}

/// Every `@tag` in `text`: the tag and its byte range (including the `@`).
pub(crate) fn tags_in(text: &str) -> Vec<(String, std::ops::Range<usize>)> {
    let mut out = Vec::new();
    let mut iter = text.char_indices().peekable();
    let mut prev: Option<char> = None;
    while let Some((i, c)) = iter.next() {
        if c == '@' && !prev.is_some_and(char::is_alphanumeric) {
            let mut end = i + 1;
            let mut word = String::new();
            while let Some(&(j, d)) = iter.peek() {
                if d.is_alphanumeric() || d == '_' || d == '-' {
                    word.push(d);
                    end = j + d.len_utf8();
                    iter.next();
                } else {
                    break;
                }
            }
            prev = word.chars().last().or(Some('@'));
            if !word.is_empty() {
                out.push((word.to_lowercase(), i..end));
            }
            continue;
        }
        prev = Some(c);
    }
    out
}

/// What the selected area and the whole picture hold, as `edit.aiFillSource` says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Holds {
    Nothing,
    Plain,
    Content,
}

impl Holds {
    pub(crate) fn parse(id: Option<&str>) -> Self {
        match id {
            Some("nothing") => Self::Nothing,
            Some("plain") => Self::Plain,
            // Unknown: treat it as something to keep in mind.
            _ => Self::Content,
        }
    }
}

/// What the image model is asked. `typed` is the person's description (may be empty), `tags`
/// the references in the order they are sent, `selection` what the selected area holds (`None`:
/// nothing is selected, and the picture goes without a mask as an instruction for the whole of
/// it). `Err` when there is nothing to go on: a blank canvas, no description and no reference,
/// or a whole-picture edit with no description.
pub(crate) fn compose(typed: &str, selection: Option<Holds>, canvas: Holds, tags: &[String]) -> Result<String, String> {
    let typed = typed.trim();
    let blank = canvas != Holds::Content;
    if typed.is_empty() && tags.is_empty() && blank {
        return Err(tl!("The canvas is blank: describe what to create, or attach a reference").into());
    }
    let mut out = String::new();
    let Some(selection) = selection else {
        if typed.is_empty() && !blank {
            return Err(tl!("Describe the change, or select the area to fill").into());
        }
        if blank {
            out.push_str("Image 1 is a blank canvas. Create a new picture on it as the description asks; compose freely and fill the whole picture.");
        } else {
            out.push_str(
                "Edit image 1 as the description asks. Change only what the description names. Keep everything else exactly as it is: the same people, faces, expressions, poses, clothes and hair, the same objects, background, lighting, colours, framing and camera angle. Do not zoom, crop, shift or reframe the picture; this is an edit of this exact photo, not a new picture.",
            );
        }
        push_references(&mut out, tags);
        out.push_str("\n\nDescription: ");
        out.push_str(&describe(typed, tags));
        return Ok(out);
    };
    if blank {
        out.push_str(
            "Image 1 is a blank canvas. Create a new picture in the area of image 1 that is transparent in the mask, as the description asks. Nothing on the canvas has to be kept; compose freely and fill the whole area.",
        );
    } else if selection == Holds::Content {
        out.push_str(
            "Edit only the area of image 1 that is transparent in the mask. What is in that area now is part of the context: keep its subject, shape, position, lighting and perspective in mind and change it only as the description asks. The result must blend seamlessly with the rest of image 1.",
        );
    } else {
        out.push_str(
            "Paint new content into the area of image 1 that is transparent in the mask, matching the lighting, perspective, colours and style of the rest of image 1 so it blends seamlessly.",
        );
    }
    out.push_str(" Keep the framing of image 1 exactly: do not zoom, crop, shift, rotate or reframe it, and leave everything outside the masked area as it is. Anything new must fit entirely inside the masked area, at a scale and perspective that match the scene.");
    push_references(&mut out, tags);
    out.push_str("\n\nDescription: ");
    if typed.is_empty() {
        out.push_str(if tags.is_empty() { "fill the area so it blends naturally with the rest of the picture." } else { "use the references in the area so they blend naturally with the picture." });
        return Ok(out);
    }
    out.push_str(&describe(typed, tags));
    Ok(out)
}

/// Which image each attached tag is.
fn push_references(out: &mut String, tags: &[String]) {
    if !tags.is_empty() {
        let list: Vec<String> = tags.iter().enumerate().map(|(i, t)| format!("image {} is @{t}", i + 2)).collect();
        out.push_str(&format!(
            " The other images are references the person attached ({}). Use what they show where the description names them; never paste them in unchanged unless asked.",
            list.join(", ")
        ));
    }
}

/// The description with each attached tag followed by its image's number.
fn describe(typed: &str, tags: &[String]) -> String {
    if typed.is_empty() {
        return "use the references so they blend naturally with the picture.".into();
    }
    // Each attached tag is followed by its image's number; `@canvas` is image 1.
    let mut described = String::new();
    let mut last = 0;
    for (word, range) in tags_in(typed) {
        let number = if word == "canvas" { Some(1) } else { tags.iter().position(|t| t.eq_ignore_ascii_case(&word)).map(|i| i + 2) };
        let Some(n) = number else { continue };
        described.push_str(typed.get(last..range.end).unwrap_or_default());
        described.push_str(&format!(" (image {n})"));
        last = range.end;
    }
    described.push_str(typed.get(last..).unwrap_or_default());
    described
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_come_from_file_names_and_never_repeat() {
        assert_eq!(tag_for("C:\\Fotos\\Mi Perro.JPG", &[]), "mi_perro");
        assert_eq!(tag_for("/tmp/hat.png", &["hat".into()]), "hat1");
        assert_eq!(tag_for("???.png", &[]), "img1");
        assert_eq!(tag_for("", &["img1".into()]), "img2");
        assert_eq!(tag_for("sombrero-rojo.webp", &[]), "sombrero-rojo");
        assert!(tag_for(&"a".repeat(200), &[]).chars().count() <= MAX_TAG);
    }

    #[test]
    fn typing_after_an_at_offers_tags() {
        assert_eq!(typing_tag("put @do", 7), Some((4, "do".into())));
        assert_eq!(typing_tag("@", 1), Some((0, String::new())));
        assert_eq!(typing_tag("put @dog here", 13), None, "a space ended it");
        assert_eq!(typing_tag("me@mail", 7), None, "not after a letter");
        assert_eq!(typing_tag("ñ @ü", 4), Some((2, "ü".into())));
        assert_eq!(typing_tag("x", 99), None);
        let (text, cursor) = complete("put @do here", 4, 2, "dog");
        assert_eq!((text.as_str(), cursor), ("put @dog here", 9));
        let (text, cursor) = complete("@", 0, 0, "hat");
        assert_eq!((text.as_str(), cursor), ("@hat ", 5));
    }

    #[test]
    fn tags_in_the_description_name_their_images() {
        let tags = vec!["dog".to_string(), "hat".to_string()];
        let p = compose("Put the hat from @hat on @Dog, keep @canvas light. Mail me@x.com @cat", Some(Holds::Content), Holds::Content, &tags).unwrap();
        assert!(p.contains("@hat (image 3) on @Dog (image 2)"), "{p}");
        assert!(p.contains("@canvas (image 1)"), "{p}");
        assert!(p.contains("me@x.com @cat"), "unknown tags and addresses stay as typed: {p}");
        assert!(p.contains("image 2 is @dog, image 3 is @hat"), "{p}");
        assert_eq!(unknown_tags("@hat @cat @cat @canvas me@x", &tags), vec!["cat".to_string()]);
    }

    #[test]
    fn the_words_follow_what_the_selection_holds() {
        let content = compose("a red umbrella", Some(Holds::Content), Holds::Content, &[]).unwrap();
        assert!(content.contains("part of the context"), "{content}");
        let empty = compose("a red umbrella", Some(Holds::Nothing), Holds::Content, &[]).unwrap();
        assert!(empty.contains("Paint new content"), "{empty}");
        let blank = compose("a red umbrella", Some(Holds::Plain), Holds::Plain, &[]).unwrap();
        assert!(blank.contains("blank canvas") && blank.contains("compose freely"), "{blank}");
        assert!(compose("  ", Some(Holds::Plain), Holds::Nothing, &[]).is_err(), "nothing to go on");
        assert!(compose("", Some(Holds::Plain), Holds::Plain, &["dog".into()]).is_ok());
        assert!(compose("", Some(Holds::Nothing), Holds::Content, &[]).unwrap().contains("blends naturally"));
        // Nothing selected on a photo: an edit of that exact photo, no mask.
        let edit = compose("make the chair hay", None, Holds::Content, &[]).unwrap();
        assert!(edit.contains("Change only what the description names") && !edit.contains("mask"), "{edit}");
        assert!(edit.ends_with("Description: make the chair hay"), "{edit}");
        assert!(compose("", None, Holds::Content, &[]).is_err(), "a whole-picture edit needs words");
        assert!(compose("a fox", None, Holds::Plain, &[]).unwrap().contains("blank canvas"));
    }

    fn png(w: u32, h: u32) -> Vec<u8> {
        let mut image = photocraft_codecs::Image::new(w, h, photocraft_codecs::ChannelLayout::Rgba, photocraft_codecs::SampleType::U8).unwrap();
        image.data_mut().fill(200);
        photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &Default::default()).unwrap()
    }

    #[test]
    fn references_are_shrunk_and_bad_files_refused() {
        let r = load("big.png", &png(3000, 1500), &[]).unwrap();
        assert_eq!(r.size, [1024, 512]);
        assert_eq!(r.tag, "big");
        let back = photocraft_codecs::decode(&r.png).unwrap();
        assert_eq!((back.width(), back.height()), (1024, 512));
        assert_eq!(load("small.png", &png(10, 20), &[]).unwrap().size, [10, 20]);
        assert!(load("x.png", b"not a picture", &[]).is_err());
        assert!(load("x.png", &[], &[]).is_err());
        let (px, w, h) = shrink(&[255; 4 * 7 * 3], 7, 3, 2);
        assert_eq!((w, h, px.len()), (2, 1, 8));
    }
}
