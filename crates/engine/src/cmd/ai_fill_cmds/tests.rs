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
fn generative_fill_says_it_needs_a_model_headless() {
    let mut s = Session::new();
    assert!(s.execute("edit.aiFillSource", json!({})).is_err(), "no document");
    let mut s = session(40, 20);
    let e = s.execute("edit.generativeFill", json!({})).unwrap_err().to_string();
    assert!(e.contains("AI image model"), "{e}");
}

#[test]
fn without_a_selection_the_whole_canvas_is_painted() {
    let mut s = session(40, 20);
    let v = s.execute("edit.aiFillSource", json!({})).unwrap();
    assert_eq!(v["region"], json!([0, 0, 40, 20]));
    let mask = decode(&v, "mask").to_rgba8();
    assert!(mask.chunks(4).all(|px| px[3] == 0), "everything may be painted");
    let png = png_of(4, 2, |_, _| [0, 0, 255, 255]);
    let v = s.execute("layer.fromAiImage", json!({"png": png})).unwrap();
    let d = s.active().unwrap();
    let layer = d.doc.layer(photocraft_doc::LayerId(v["layer"].as_u64().unwrap())).unwrap();
    assert!(layer.mask.is_none(), "no selection, no mask");
    assert!(layer.surface().unwrap().sample_channel(39, 19, 2) > 0.99);
}

#[test]
fn a_small_selection_is_sent_with_context_around_it() {
    let canvas = Rect::new(0, 0, 6000, 4000);
    // 200 × 100 at (3000, 2000): 120 px of context (60 % of 200) on every side, then widened to
    // at most 2.5:1.
    let r = fill_region(canvas, Some(Rect::new(3000, 2000, 3200, 2100)));
    assert!(r.contains_rect(&Rect::new(2880, 1880, 3320, 2220)), "{r:?}");
    assert!(f64::from(r.width()) / f64::from(r.height()) <= 2.5 + 1e-9);
    // Near a corner the region slides inside the canvas instead of losing its context.
    let r = fill_region(canvas, Some(Rect::new(0, 0, 100, 100)));
    assert_eq!((r.x0, r.y0), (0, 0));
    assert!(r.width() >= 290 && r.height() >= 290, "{r:?}");
    assert_eq!(fill_region(canvas, None), canvas);
    assert_eq!(fill_region(canvas, Some(Rect::new(9000, 9000, 9100, 9100))), canvas, "off the canvas");
    // The engine sends only that region, and the result goes back over it.
    let mut s = session(1000, 600);
    s.execute("select.rect", json!({"x": 500, "y": 300, "width": 50, "height": 50})).unwrap();
    let v = s.execute("edit.aiFillSource", json!({})).unwrap();
    assert_eq!(v["region"], json!([404, 204, 242, 242]));
    let png = png_of(4, 4, |_, _| [255, 0, 0, 255]);
    let v = s.execute("layer.fromAiImage", json!({"png": png, "region": v["region"]})).unwrap();
    let d = s.active().unwrap();
    let surface = d.doc.layer(photocraft_doc::LayerId(v["layer"].as_u64().unwrap())).unwrap().surface().unwrap();
    assert!(surface.sample_channel(520, 320, 0) > 0.99, "painted in the region");
    assert!(surface.sample_channel(100, 100, 3) < 0.01, "nothing outside it");
    let full = s.execute("edit.aiFillSource", json!({"crop": false})).unwrap();
    assert_eq!(full["region"], json!([0, 0, 1000, 600]));
}

#[test]
fn bad_regions_are_refused() {
    let mut s = session(10, 10);
    let png = png_of(2, 2, |_, _| [0, 0, 0, 255]);
    for region in [json!([0, 0, 0, 5]), json!([0, 0, 20, 5]), json!([-1, 0, 5, 5]), json!([1, 2]), json!("x"), json!([0, 0, 1e12, 5])] {
        assert!(s.execute("layer.fromAiImage", json!({"png": png, "region": region})).is_err(), "{region}");
    }
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

#[test]
fn the_fill_source_says_what_the_selection_and_the_canvas_hold() {
    // A new document is one plain colour: blank, so the model may create freely.
    let mut s = session(64, 32);
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 32, "height": 32})).unwrap();
    let v = s.execute("edit.aiFillSource", json!({})).unwrap();
    assert_eq!((v["selection"].as_str(), v["canvas"].as_str()), (Some("plain"), Some("plain")));
    // Stripes on the left half only: the selection over them holds content, the right does not.
    s.execute("select.all", json!({})).unwrap();
    let png = png_of(64, 32, |x, y| if x < 32 { if (x + y) % 4 < 2 { [0, 0, 0, 255] } else { [255, 255, 0, 255] } } else { [0, 0, 0, 0] });
    s.execute("layer.fromAiImage", json!({"png": png})).unwrap();
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 32, "height": 32})).unwrap();
    let v = s.execute("edit.aiFillSource", json!({"sampleAllLayers": false})).unwrap();
    assert_eq!((v["selection"].as_str(), v["canvas"].as_str()), (Some("content"), Some("content")));
    s.execute("select.rect", json!({"x": 32, "y": 0, "width": 32, "height": 32})).unwrap();
    let v = s.execute("edit.aiFillSource", json!({"sampleAllLayers": false})).unwrap();
    assert_eq!(v["selection"].as_str(), Some("nothing"), "transparent on the active layer");
}

#[test]
fn the_new_picture_is_brought_back_to_the_originals_colours() {
    // Grey 0.5 outside the selection; the model returned it at 0.7 with a bright patch inside.
    let (n, inside) = (4000, 1000..1500);
    let original = vec![[0.5, 0.4, 0.3, 1.0]; n];
    let coverage: Vec<f32> = (0..n).map(|i| if inside.contains(&i) { 1.0 } else { 0.0 }).collect();
    let mut data: Vec<f32> = (0..n).flat_map(|i| if inside.contains(&i) { [0.9, 0.9, 0.9, 1.0] } else { [0.7, 0.6, 0.5, 1.0] }).collect();
    match_colors(&mut data, &original, &coverage);
    assert!((data[0] - 0.5).abs() < 1e-3 && (data[1] - 0.4).abs() < 1e-3, "{:?}", &data[..4]);
    assert!(data[1000 * 4] > 0.6, "the new content keeps its own, corrected, colour");
    // Too few pixels outside: nothing changes.
    let mut few = vec![0.7f32; 40];
    match_colors(&mut few, &[[0.5, 0.5, 0.5, 1.0]; 10], &[0.0; 10]);
    assert!(few.iter().all(|v| (*v - 0.7).abs() < 1e-6));
}

#[test]
fn an_edit_of_the_whole_picture_keeps_only_what_changed() {
    // Grey canvas; the model returned the same grey with a red square at (40..60, 40..60).
    let (w, h) = (100usize, 100usize);
    let original = vec![[0.5, 0.5, 0.5, 1.0]; w * h];
    let data: Vec<f32> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            if (40..60).contains(&x) && (40..60).contains(&y) { [1.0, 0.0, 0.0, 1.0] } else { [0.51, 0.5, 0.49, 1.0] }
        })
        .collect();
    let m = change_mask(&data, &original, w, h);
    assert!(m[50 * w + 50] > 0.99, "the change is kept");
    assert!(m[40 * w + 40] > 0.5, "its edge too");
    assert!(m[5 * w + 5] < 0.01, "the unchanged picture is not");
    // Through the command: a layer masked to the change, no selection needed.
    let mut s = session(100, 100);
    let png = png_of(100, 100, |x, y| if (40..60).contains(&x) && (40..60).contains(&y) { [255, 0, 0, 255] } else { [255, 255, 255, 255] });
    let v = s.execute("layer.fromAiImage", json!({"png": png, "keep": "changes"})).unwrap();
    let d = s.active().unwrap();
    let layer = d.doc.layer(photocraft_doc::LayerId(v["layer"].as_u64().unwrap())).unwrap();
    let mask = layer.mask.as_ref().expect("masked to what changed");
    assert!(mask.surface.sample_channel(50, 50, 0) > 0.99);
    assert!(mask.surface.sample_channel(5, 5, 0) < 0.01);
    assert!(s.execute("layer.fromAiImage", json!({"png": png_of(2, 2, |_, _| [0, 0, 0, 255]), "keep": "some"})).is_err());
}
