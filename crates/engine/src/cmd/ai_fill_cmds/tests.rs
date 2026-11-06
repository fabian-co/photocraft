use super::*;

fn session(w: u32, h: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": w, "height": h})).unwrap();
    s
}

fn decode(v: &Value, key: &str) -> photocraft_codecs::Image {
    let bytes = base64::engine::general_purpose::STANDARD.decode(v[key].as_str().unwrap()).unwrap();
    photocraft_codecs::decode(&bytes).unwrap()
}

fn png_of(w: u32, h: u32, px: impl Fn(u32, u32) -> [u8; 4]) -> String {
    let mut image = photocraft_codecs::Image::new(w, h, photocraft_codecs::ChannelLayout::Rgba, photocraft_codecs::SampleType::U8).unwrap();
    let data = image.data_mut();
    for y in 0..h {
        for x in 0..w {
            let at = ((y * w + x) * 4) as usize;
            data[at..at + 4].copy_from_slice(&px(x, y));
        }
    }
    let png = photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &Default::default()).unwrap();
    base64::engine::general_purpose::STANDARD.encode(png)
}

#[test]
fn generative_fill_needs_a_selection_and_says_it_needs_a_model_headless() {
    let mut s = session(40, 20);
    assert!(s.execute("edit.generativeFill", json!({})).is_err());
    assert!(s.execute("edit.aiFillSource", json!({})).is_err());
    s.execute("select.all", json!({})).unwrap();
    let e = s.execute("edit.generativeFill", json!({})).unwrap_err().to_string();
    assert!(e.contains("AI image model"), "{e}");
}

#[test]
fn the_fill_source_masks_the_selection_as_transparent() {
    let mut s = session(40, 20);
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 20, "height": 20})).unwrap();
    // 40 × 20 rounds to the nearest multiples of 16: 48 × 16.
    let v = s.execute("edit.aiFillSource", json!({})).unwrap();
    assert_eq!((v["width"].as_u64(), v["height"].as_u64()), (Some(48), Some(16)));
    let picture = decode(&v, "png");
    let mask = decode(&v, "mask").to_rgba8();
    assert_eq!((picture.width(), picture.height()), (48, 16));
    let alpha = |x: usize, y: usize| mask[(y * 48 + x) * 4 + 3];
    assert_eq!(alpha(6, 8), 0, "selected: transparent");
    assert_eq!(alpha(42, 8), 255, "outside: opaque");
}

#[test]
fn an_asked_size_is_rounded_to_multiples_of_16() {
    let mut s = session(300, 200);
    s.execute("select.all", json!({})).unwrap();
    let v = s.execute("edit.aiFillSource", json!({"width": 1000, "height": 700})).unwrap();
    assert_eq!((v["width"].as_u64(), v["height"].as_u64()), (Some(1008), Some(704)));
    let picture = decode(&v, "png");
    assert_eq!((picture.width(), picture.height()), (1008, 704));
    // A large canvas is scaled to the longest side first, then rounded.
    let mut s = session(3000, 1999);
    s.execute("select.all", json!({})).unwrap();
    let v = s.execute("edit.aiFillSource", json!({})).unwrap();
    assert_eq!((v["width"].as_u64(), v["height"].as_u64()), (Some(1536), Some(1024)));
    assert!(s.execute("edit.aiFillSource", json!({"width": 512})).is_err(), "one side alone is refused");
}

#[test]
fn sides_round_to_the_nearest_step() {
    assert_eq!(round_side(1.0), 16);
    assert_eq!(round_side(23.9), 16);
    assert_eq!(round_side(24.0), 32);
    assert_eq!(round_side(1023.0), 1024);
    assert_eq!(round_side(f64::NAN), 16);
    assert_eq!(round_side(1e9), 4096);
}

#[test]
fn the_returned_picture_becomes_one_masked_layer_and_undoes() {
    let mut s = session(40, 20);
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 20, "height": 20})).unwrap();
    let before = s.active().unwrap().doc.layers.len();
    let png = png_of(4, 2, |_, _| [255, 0, 0, 255]);
    let v = s.execute("layer.fromAiImage", json!({"png": png, "name": "Fill"})).unwrap();
    let d = s.active().unwrap();
    assert_eq!(d.doc.layers.len(), before + 1);
    let id = photocraft_doc::LayerId(v["layer"].as_u64().unwrap());
    assert_eq!(d.active_layer, Some(id));
    let layer = d.doc.layer(id).unwrap();
    assert_eq!(layer.name, "Fill");
    let surface = layer.surface().unwrap();
    assert!(surface.sample_channel(30, 10, 0) > 0.99, "red over the whole canvas");
    let mask = layer.mask.as_ref().expect("masked by the selection");
    assert!(mask.surface.sample_channel(5, 10, 0) > 0.99);
    assert!(mask.surface.sample_channel(35, 10, 0) < 0.01);
    s.undo();
    assert_eq!(s.active().unwrap().doc.layers.len(), before);
}

#[test]
fn bad_pictures_are_refused() {
    let mut s = session(10, 10);
    s.execute("select.all", json!({})).unwrap();
    assert!(s.execute("layer.fromAiImage", json!({})).is_err());
    assert!(s.execute("layer.fromAiImage", json!({"png": "not base64!"})).is_err());
    assert!(s.execute("layer.fromAiImage", json!({"png": "aGVsbG8="})).is_err());
    let png = png_of(2, 2, |_, _| [0, 0, 0, 255]);
    assert!(s.execute("layer.fromAiImage", json!({"png": png, "document": 999_999})).is_err());
}
