//! What one Generative Fill is likely to cost, shown in the task bar before Generate.
//!
//! OpenAI bills its image models by tokens: the picture it makes (output tokens, which grow with
//! the quality and the number of pixels) plus what it reads (the description, the picture, and
//! every reference image). The figures below are OpenAI's published rates and its token counts
//! for a 1024 × 1024 picture, scaled by area; DALL·E 2 is billed per picture. They are an estimate
//! for a person deciding, never a bill: OpenAI's own usage is what is charged. A model not listed
//! here gets no estimate rather than a guess.

use super::task_bar::Quality;

/// US dollars for one million tokens.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rates {
    text_in: f64,
    image_in: f64,
    image_out: f64,
}

/// How a model is billed.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Billing {
    /// By tokens: `tokens(quality)` output tokens for a 1024 × 1024 picture.
    Tokens { rates: Rates, family: Family },
    /// A fixed price per picture by its size (DALL·E 2).
    PerPicture,
}

/// Which token counts a model uses per quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    /// gpt-image-1, 1-mini, 1.5 and chatgpt-image.
    One,
    /// gpt-image-2.
    Two,
    /// gpt-image-2.5 and later, whose tiers were renamed and gained Extra high and Maximum.
    TwoFive,
}

impl Family {
    /// Output tokens for a 1024 × 1024 picture at `quality` (`None`: a quality the model lacks).
    fn tokens(self, quality: Quality) -> Option<f64> {
        Some(match (self, quality) {
            (Family::One, Quality::Low) => 272.0,
            (Family::One, Quality::Medium) => 1056.0,
            (Family::One, Quality::High) => 4160.0,
            (Family::Two | Family::TwoFive, Quality::Low) => 196.0,
            (Family::Two, Quality::Medium) => 1756.0,
            (Family::Two, Quality::High) => 7024.0,
            (Family::TwoFive, Quality::Medium) => 439.0,
            (Family::TwoFive, Quality::High) => 1756.0,
            (Family::TwoFive, Quality::XHigh) => 3122.0,
            (Family::TwoFive, Quality::Max) => 7024.0,
            _ => return None,
        })
    }

    /// The cheapest and dearest quality Automatic may pick.
    fn auto_range(self) -> (Quality, Quality) {
        (Quality::Low, Quality::High)
    }
}

fn billing(model: &str) -> Option<Billing> {
    let lower = model.to_ascii_lowercase();
    let tokens = |text_in, image_in, image_out, family| Some(Billing::Tokens { rates: Rates { text_in, image_in, image_out }, family });
    match super::task_bar::gpt_image_version(&lower) {
        Some(v) if v >= (2, 5) => tokens(5.0, 8.0, 30.0, Family::TwoFive),
        Some(v) if v >= (2, 0) => tokens(5.0, 8.0, 30.0, Family::Two),
        Some(_) if lower.contains("mini") => tokens(2.0, 2.5, 8.0, Family::One),
        Some(v) if v >= (1, 5) => tokens(5.0, 8.0, 32.0, Family::One),
        Some(_) => tokens(5.0, 10.0, 40.0, Family::One),
        None if lower.contains("chatgpt-image") => tokens(5.0, 8.0, 32.0, Family::One),
        None if lower.contains("dall-e-2") => Some(Billing::PerPicture),
        None => None,
    }
}

/// What one picture of `w` × `h` costs to read: OpenAI's tile count (the picture fitted in 2048 ×
/// 2048, then its short side brought to 768, in 512-pixel tiles).
fn input_tokens(w: u32, h: u32) -> f64 {
    let (mut w, mut h) = (f64::from(w.max(1)), f64::from(h.max(1)));
    let fit = (2048.0 / w.max(h)).min(1.0);
    (w, h) = (w * fit, h * fit);
    let short = (768.0 / w.min(h)).min(1.0);
    (w, h) = (w * short, h * short);
    let tiles = (w / 512.0).ceil() * (h / 512.0).ceil();
    85.0 + 170.0 * tiles
}

/// An estimate in US dollars: `low` to `high` (equal unless the quality is Automatic).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Estimate {
    pub low: f64,
    pub high: f64,
}

/// What goes into one request besides the model and the quality.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Order {
    /// The picture's size (and so the result's).
    pub size: [u32; 2],
    /// Reference images, each with its size.
    pub references: usize,
    pub reference_side: u32,
    /// Characters of the description the model reads.
    pub prompt_chars: usize,
}

/// The likely cost of one Generative Fill with `model` at `quality` (`None`: the model takes no
/// quality). `None` when the model's prices are not known.
/// With `learned` output tokens per 1024 × 1024 (what the last generation with this model and
/// quality really used), that replaces the published figure.
pub(crate) fn estimate_with(model: &str, quality: Option<Quality>, order: Order, learned: Option<f64>) -> Option<Estimate> {
    let [w, h] = order.size;
    match billing(model)? {
        Billing::PerPicture => {
            let side = w.max(h);
            let usd = if side <= 256 {
                0.016
            } else if side <= 512 {
                0.018
            } else {
                0.020
            };
            Some(Estimate { low: usd, high: usd })
        }
        Billing::Tokens { rates, family } => {
            let area = f64::from(w.max(1)) * f64::from(h.max(1)) / (1024.0 * 1024.0);
            let reads = input_tokens(w, h) + order.references as f64 * input_tokens(order.reference_side, order.reference_side);
            // About four characters a token, plus the instructions this app adds.
            let words = (order.prompt_chars as f64 + 600.0) / 4.0;
            let fixed = (reads * rates.image_in + words * rates.text_in) / 1e6;
            let made = |q: Quality| family.tokens(q).map(|t| t * area * rates.image_out / 1e6 + fixed);
            if let Some(per) = learned.filter(|t| t.is_finite() && *t > 0.0) {
                let one = per * area * rates.image_out / 1e6 + fixed;
                return Some(Estimate { low: one, high: one });
            }
            let (low, high) = match quality {
                Some(Quality::Auto) | None => {
                    let (a, b) = family.auto_range();
                    (made(a)?, made(b)?)
                }
                Some(q) => {
                    let one = made(q)?;
                    (one, one)
                }
            };
            Some(Estimate { low, high })
        }
    }
}

/// What a generation that used `input` and `output` tokens cost, in US dollars. Input is billed
/// at the picture rate (most of it is pictures; the description is a few hundred tokens).
pub(crate) fn actual(model: &str, input: u64, output: u64) -> Option<f64> {
    match billing(model)? {
        Billing::Tokens { rates, .. } => Some((input as f64 * rates.image_in + output as f64 * rates.image_out) / 1e6),
        Billing::PerPicture => None,
    }
}

/// `$0.053 USD`.
pub(crate) fn dollars_label(usd: f64) -> String {
    format!("{} USD", dollars(usd))
}

fn dollars(usd: f64) -> String {
    if usd < 0.001 {
        "< $0.001".into()
    } else if usd < 0.1 {
        format!("${usd:.3}")
    } else {
        format!("${usd:.2}")
    }
}

/// `≈ $0.053` or `≈ $0.006 – $0.21`.
pub(crate) fn label(e: Estimate) -> String {
    if (e.high - e.low).abs() < 0.0005 { format!("≈ {} USD", dollars(e.low)) } else { format!("≈ {} – {} USD", dollars(e.low), dollars(e.high)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimate(model: &str, quality: Option<Quality>, order: Order) -> Option<Estimate> {
        estimate_with(model, quality, order, None)
    }

    fn order(w: u32, h: u32) -> Order {
        Order { size: [w, h], references: 0, reference_side: 1024, prompt_chars: 0 }
    }

    /// `a` is the published output price `b` plus what reading the picture and the words adds.
    fn close(a: f64, b: f64) -> bool {
        a >= b - 0.001 && a < b + 0.012
    }

    #[test]
    fn square_pictures_cost_what_openai_publishes() {
        let one = |m: &str, q| estimate(m, Some(q), order(1024, 1024)).unwrap().low;
        assert!(close(one("gpt-image-1", Quality::Low), 0.011), "{}", one("gpt-image-1", Quality::Low));
        assert!(close(one("gpt-image-1", Quality::Medium), 0.042));
        assert!(close(one("gpt-image-1", Quality::High), 0.167));
        assert!(close(one("gpt-image-2", Quality::Medium), 0.053));
        assert!(close(one("gpt-image-2", Quality::High), 0.211));
        assert!(close(one("gpt-image-2.5-flare", Quality::High), 0.053));
        assert!(close(one("gpt-image-2.5-sunburst", Quality::Max), 0.211));
        assert!(one("gpt-image-1-mini", Quality::High) < one("gpt-image-1", Quality::High) / 4.0);
        assert_eq!(estimate("dall-e-2", None, order(512, 512)), Some(Estimate { low: 0.018, high: 0.018 }));
        assert_eq!(estimate("flux-fill", Some(Quality::High), order(1024, 1024)), None, "no guess");
    }

    #[test]
    fn cost_grows_with_size_quality_and_references() {
        let at = |q, o| estimate("gpt-image-2", Some(q), o).unwrap().low;
        assert!(at(Quality::High, order(2048, 2048)) > 3.0 * at(Quality::High, order(1024, 1024)));
        assert!(at(Quality::High, order(1024, 1024)) > at(Quality::Medium, order(1024, 1024)));
        let with = Order { references: 3, ..order(1024, 1024) };
        assert!(at(Quality::Low, with) > at(Quality::Low, order(1024, 1024)));
        let auto = estimate("gpt-image-2", Some(Quality::Auto), order(1024, 1024)).unwrap();
        assert!(auto.low < auto.high);
        assert_eq!(estimate("gpt-image-1", Some(Quality::Max), order(1024, 1024)), None, "a quality the model lacks");
    }

    #[test]
    fn real_usage_gives_the_cost_and_tunes_the_estimate() {
        // gpt-image-2.6-sunburst at low made 903 tokens for about 1.5 megapixels.
        let usd = actual("gpt-image-2.6-sunburst", 2581, 903).unwrap();
        assert!((usd - (2581.0 * 8.0 + 903.0 * 30.0) / 1e6).abs() < 1e-9);
        assert_eq!(actual("dall-e-2", 1, 1), None);
        assert_eq!(actual("flux", 1, 1), None);
        let base = estimate("gpt-image-2.6-sunburst", Some(Quality::Low), order(1024, 1024)).unwrap();
        let tuned = estimate_with("gpt-image-2.6-sunburst", Some(Quality::Low), order(1024, 1024), Some(600.0)).unwrap();
        assert!(tuned.low > base.low && (tuned.low - tuned.high).abs() < 1e-12);
    }

    #[test]
    fn labels_round_for_people() {
        assert_eq!(label(Estimate { low: 0.0531, high: 0.0531 }), "≈ $0.053 USD");
        assert_eq!(label(Estimate { low: 0.006, high: 0.211 }), "≈ $0.006 – $0.21 USD");
        assert_eq!(label(Estimate { low: 0.0004, high: 0.0004 }), "≈ < $0.001 USD");
    }
}
