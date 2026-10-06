//! Lumetri Color as one evaluated op ([`crate::gpufx::FxOp::Lumetri`]), so the GPU effect stage
//! can grade a layer instead of handing it to the CPU.
//!
//! [`LumetriOp::eval`] reads the effect's parameters into plain numbers and curve tables exactly
//! as `effects::lumetri` does, and [`LumetriOp::pixel`] repeats its arithmetic per pixel: the
//! Basic Correction / Creative / Vignette pass, the Curves / Wheels / Look pass, then the HSL
//! Secondary key and correction, each ending in the grade space's decode like the three CPU
//! passes. `fx.wgsl` mirrors `pixel` line by line; [`LumetriOp::gpu_data`] is the table it reads.
//!
//! Covered: SDR (Rec. 709) grading without LUT files, Creative Sharpen or HSL Denoise / Blur.
//! `eval` returns None for anything else, and the effect renders on the CPU as before.

use std::sync::Arc;

use filmcraft_color::{GradeSpace, hsl_to_rgb, linear_to_srgb, rgb_to_hsl, srgb_to_linear};
use filmcraft_project::EffectInstance;

use crate::effects::{FxCtx, apply_look, b, choice, color, curve_lut, curve_param, f, hsl_key, hue_lut, is_identity_curve, on, text, wheel_rgb};
use crate::image::Image;

/// Entries of a curve table (the CPU's `lumetri_advanced` LUT size).
pub const CURVE_N: usize = 1024;

/// Curve tables, in [`LumetriOp::curves`] order.
pub const CURVES: [&str; 9] = ["curve_luma", "curve_red", "curve_green", "curve_blue", "hue_vs_sat", "hue_vs_hue", "hue_vs_luma", "luma_vs_sat", "sat_vs_sat"];

/// HSL Secondary, evaluated.
#[derive(Clone, Debug, PartialEq)]
pub struct LumetriHsl {
    pub hue: f32,
    pub range: f32,
    pub sat_min: f32,
    pub luma_min: f32,
    pub luma_max: f32,
    pub soft: f32,
    pub show_mask: u32,
    pub temp: f32,
    pub tint: f32,
    pub sat: f32,
    pub shift: f32,
}

/// Lumetri Color, evaluated for one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct LumetriOp {
    // Basic Correction, Creative (Faded Film, split tone, saturation, vibrance), Vignette
    pub gains: [f32; 3],
    pub exposure: f32,
    pub b0: f32,
    pub w0: f32,
    pub shadows: f32,
    pub highlights: f32,
    pub contrast: f32,
    pub faded: f32,
    pub creative: bool,
    pub shadow_tint: [f32; 3],
    pub highlight_tint: [f32; 3],
    pub sat: f32,
    pub vibrance: f32,
    pub vignette: f32,
    pub v_mid: f32,
    pub v_round: f32,
    pub v_feather: f32,
    // Curves, Color Wheels, Look (skipped as a whole unless `advanced`)
    pub advanced: bool,
    pub look: u32,
    pub look_k: f32,
    pub wheels: bool,
    pub wheel: [[f32; 3]; 3],
    pub lightness: [f32; 3],
    /// [`CURVES`] tables (None: the curve is off or does nothing).
    pub curves: [Option<Arc<Vec<f32>>>; 9],
    pub hsl: Option<LumetriHsl>,
}

impl LumetriOp {
    /// Evaluate `e` at `cx.t`. None when the grade needs something only the CPU does (HDR, LUT
    /// files, Creative Sharpen, HSL Denoise / Blur).
    pub fn eval(e: &EffectInstance, cx: &FxCtx) -> Option<LumetriOp> {
        let (basic_on, creative_on, vignette_on) = (on(e, "basic_on"), on(e, "creative_on"), on(e, "vignette_on"));
        let (curves_on, wheels_on) = (on(e, "curves_on"), on(e, "wheels_on"));
        if GradeSpace::new(cx.working, f(e, "hdr_white", cx)).is_hdr() || GradeSpace::new(cx.working, f(e, "curves_hdr_range", cx)).is_hdr() {
            return None;
        }
        if basic_on && !text(e, "input_lut").is_empty() {
            return None;
        }
        if creative_on && (!text(e, "look_lut").is_empty() || (f(e, "sharpen", cx) / 100.0).abs() > 1e-3) {
            return None;
        }
        let hsl_on = b(e, "hsl_on");
        if hsl_on && (f(e, "hsl_denoise", cx).clamp(0.0, 100.0) > 0.0 || f(e, "hsl_blur", cx).clamp(0.0, 100.0) > 0.0) {
            return None;
        }
        let bf = |id: &str| if basic_on { f(e, id, cx) } else { 0.0 };
        let (temp, tint) = (bf("temperature") / 100.0, bf("tint") / 100.0);
        let (wh, bl) = (bf("whites") / 100.0, bf("blacks") / 100.0);
        let sat = if basic_on { f(e, "saturation", cx) / 100.0 } else { 1.0 } * if creative_on { f(e, "creative_sat", cx) / 100.0 } else { 1.0 };
        let st = color(e, "shadow_tint", cx);
        let ht = color(e, "highlight_tint", cx);

        let lut = |id: &str| curve_param(e, id).filter(|c| curves_on && !is_identity_curve(c)).map(|c| Arc::new(curve_lut(&c, CURVE_N)));
        let hue = |id: &str| curve_param(e, id).filter(|_| curves_on).and_then(|c| hue_lut(&c, CURVE_N)).map(Arc::new);
        let curves =
            [lut(CURVES[0]), lut(CURVES[1]), lut(CURVES[2]), lut(CURVES[3]), hue(CURVES[4]), hue(CURVES[5]), hue(CURVES[6]), hue(CURVES[7]), hue(CURVES[8])];
        let look = if creative_on { choice(e, "look") } else { 0 };
        let v2 = |id: &str| e.param(id).map(|p| p.vec2_at(cx.t)).unwrap_or_default();
        let wheel = [wheel_rgb(v2("wheel_shadows")), wheel_rgb(v2("wheel_midtones")), wheel_rgb(v2("wheel_highlights"))];
        let lightness = [f(e, "wheel_shadows_l", cx) / 100.0, f(e, "wheel_midtones_l", cx) / 100.0, f(e, "wheel_highlights_l", cx) / 100.0];
        let wheels = wheels_on && (wheel.iter().flatten().any(|v| v.abs() > 1e-5) || lightness.iter().map(|v| v.abs()).sum::<f32>() > 1e-5);
        let advanced = curves.iter().any(Option::is_some) || look > 0 || wheels;

        let hsl = hsl_on.then(|| LumetriHsl {
            hue: f(e, "hsl_hue", cx) / 360.0,
            range: (f(e, "hsl_hue_range", cx) / 360.0).max(1e-3),
            sat_min: f(e, "hsl_sat_min", cx) / 100.0,
            luma_min: f(e, "hsl_luma_min", cx) / 100.0,
            luma_max: f(e, "hsl_luma_max", cx) / 100.0,
            soft: (f(e, "hsl_soft", cx) / 100.0 * 0.3).max(0.01),
            show_mask: choice(e, "hsl_show_mask"),
            temp: f(e, "hsl_temp", cx) / 100.0,
            tint: f(e, "hsl_tint", cx) / 100.0,
            sat: f(e, "hsl_sat", cx) / 100.0,
            shift: f(e, "hsl_hue_shift", cx) / 360.0,
        });

        Some(LumetriOp {
            gains: [1.0 + 0.35 * temp, 1.0 - 0.3 * tint, 1.0 - 0.35 * temp],
            exposure: 2f32.powf(bf("exposure")),
            b0: -bl * 0.15,
            w0: 1.0 - wh * 0.15,
            shadows: bf("shadows") / 100.0,
            highlights: bf("highlights") / 100.0,
            contrast: bf("contrast") / 100.0,
            faded: if creative_on { f(e, "faded_film", cx) / 100.0 } else { 0.0 },
            creative: creative_on,
            shadow_tint: [st[0], st[1], st[2]],
            highlight_tint: [ht[0], ht[1], ht[2]],
            sat,
            vibrance: if creative_on { f(e, "vibrance", cx) / 100.0 } else { 0.0 },
            vignette: if vignette_on { f(e, "vignette_amount", cx) } else { 0.0 },
            v_mid: f(e, "vignette_midpoint", cx) / 100.0,
            v_round: f(e, "vignette_roundness", cx) / 100.0,
            v_feather: f(e, "vignette_feather", cx) / 100.0,
            advanced,
            look,
            look_k: f(e, "look_intensity", cx) / 100.0,
            wheels,
            wheel,
            lightness,
            curves,
            hsl,
        })
    }

    /// Every number finite (the GPU leaves NaN / infinity behaviour to the CPU).
    pub fn finite(&self) -> bool {
        let scalars = [
            self.exposure,
            self.b0,
            self.w0,
            self.shadows,
            self.highlights,
            self.contrast,
            self.faded,
            self.sat,
            self.vibrance,
            self.vignette,
            self.v_mid,
            self.v_round,
            self.v_feather,
            self.look_k,
        ];
        let hsl = self
            .hsl
            .as_ref()
            .is_none_or(|h| [h.hue, h.range, h.sat_min, h.luma_min, h.luma_max, h.soft, h.temp, h.tint, h.sat, h.shift].iter().all(|v| v.is_finite()));
        scalars
            .iter()
            .chain(&self.gains)
            .chain(&self.shadow_tint)
            .chain(&self.highlight_tint)
            .chain(self.wheel.iter().flatten())
            .chain(&self.lightness)
            .all(|v| v.is_finite())
            && self.curves.iter().flatten().all(|c| c.iter().all(|v| v.is_finite()))
            && hsl
    }

    /// The CPU reference on a working image (straight colour per pixel, like `Image::map_rgb`).
    pub fn apply(&self, img: &mut Image) {
        let (w, h) = (img.w as f32, img.h as f32);
        img.map_rgb(|c, x, y| self.pixel(c, x as f32, y as f32, w, h));
    }

    /// One pixel: linear straight colour at (`x`, `y`) of a `w`×`h` working image.
    pub fn pixel(&self, c: [f32; 3], x: f32, y: f32, w: f32, h: f32) -> [f32; 3] {
        let c = self.basic(c, x, y, w, h);
        let c = if self.advanced { self.advanced_pass(c) } else { c };
        match &self.hsl {
            Some(hsl) => secondary(hsl, c),
            None => c,
        }
    }

    fn basic(&self, c: [f32; 3], x: f32, y: f32, w: f32, h: f32) -> [f32; 3] {
        let g = self.gains;
        let lin = [c[0] * g[0] * self.exposure, c[1] * g[1] * self.exposure, c[2] * g[2] * self.exposure];
        let mut v = enc(lin);
        v = v.map(|q| (q - self.b0) / (self.w0 - self.b0).max(1e-3));
        let l = luma(v);
        let ws = (1.0 - l).clamp(0.0, 1.0).powi(3);
        let whl = l.clamp(0.0, 1.0).powi(3);
        let nl = (l + self.shadows * 0.35 * ws + self.highlights * 0.35 * whl).max(0.0);
        if l > 1e-5 {
            let k = nl / l;
            v = v.map(|q| q * k);
        }
        if self.contrast.abs() > 1e-4 {
            let k = 1.0 + self.contrast;
            v = v.map(|q| {
                let q = q.clamp(0.0, 1.0);
                let s = q * q * (3.0 - 2.0 * q);
                if k >= 1.0 { q + (s - q) * (k - 1.0) } else { 0.5 + (q - 0.5) * k }
            });
        }
        if self.faded > 0.0 {
            v = v.map(|q| q * (1.0 - 0.25 * self.faded) + 0.12 * self.faded);
        }
        if self.creative {
            let l2 = luma(v).clamp(0.0, 1.0);
            for k in 0..3 {
                v[k] += (self.shadow_tint[k] - 0.5) * 0.3 * (1.0 - l2) + (self.highlight_tint[k] - 0.5) * 0.3 * l2;
            }
        }
        let l3 = luma(v);
        let cur_sat = v[0].max(v[1]).max(v[2]) - v[0].min(v[1]).min(v[2]);
        let s = self.sat * (1.0 + self.vibrance * (1.0 - cur_sat.clamp(0.0, 1.0)));
        v = v.map(|q| l3 + (q - l3) * s);
        let va = self.vignette;
        if va.abs() > 1e-4 {
            let aspect = w / h;
            let nx = (x / w - 0.5) * 2.0 * if self.v_round < 0.0 { aspect.powf(-self.v_round) } else { 1.0 };
            let ny = (y / h - 0.5) * 2.0;
            let d = (nx * nx + ny * ny).sqrt() / std::f32::consts::SQRT_2;
            let edge = ((d - self.v_mid * 0.9) / (self.v_feather.max(0.01) * 0.9)).clamp(0.0, 1.0);
            let e2 = edge * edge * (3.0 - 2.0 * edge);
            let k = 1.0 + va * 0.2 * e2;
            v = v.map(|q| if va < 0.0 { q * k.max(0.0) } else { q + (1.0 - q) * (k - 1.0) });
        }
        dec(v)
    }

    fn advanced_pass(&self, c: [f32; 3]) -> [f32; 3] {
        let mut v = enc(c);
        if self.look > 0 {
            let lk = apply_look(self.look, v);
            let k = self.look_k;
            v = [v[0] + (lk[0] - v[0]) * k, v[1] + (lk[1] - v[1]) * k, v[2] + (lk[2] - v[2]) * k];
        }
        if self.wheels {
            let [ws, wm, wh] = self.wheel;
            let [ls, lm, lh] = self.lightness;
            let l = luma(v).clamp(0.0, 1.0);
            let wsh = (1.0 - l).powi(2);
            let whi = l * l;
            let wmid = (1.0 - wsh - whi).max(0.0);
            for k in 0..3 {
                v[k] += (ws[k] * 0.3 + ls * 0.3) * wsh;
                v[k] += (wm[k] * 0.3 + lm * 0.3) * wmid;
                v[k] *= 1.0 + (wh[k] * 0.5 + lh * 0.5) * whi;
            }
        }
        let [luma_c, red, green, blue, hvs, hvh, hvl, lvs, svs] = &self.curves;
        if let Some(t) = luma_c {
            v = v.map(|q| sample(t, q));
        }
        if let Some(t) = red {
            v[0] = sample(t, v[0]);
        }
        if let Some(t) = green {
            v[1] = sample(t, v[1]);
        }
        if let Some(t) = blue {
            v[2] = sample(t, v[2]);
        }
        if hvs.is_some() || hvh.is_some() || hvl.is_some() || lvs.is_some() || svs.is_some() {
            let u = v.map(|q| q.clamp(0.0, 1.0));
            let mut hh = rgb_to_hsl(u[0], u[1], u[2]);
            let (h0, s0, l0) = (hh[0], hh[1], hh[2]);
            if let Some(t) = hvh {
                hh[0] = (hh[0] + (sample(t, h0) - 0.5)).rem_euclid(1.0);
            }
            let mut sm = 1.0;
            if let Some(t) = hvs {
                sm *= sample(t, h0) * 2.0;
            }
            if let Some(t) = lvs {
                sm *= sample(t, l0) * 2.0;
            }
            if let Some(t) = svs {
                sm *= sample(t, s0) * 2.0;
            }
            hh[1] = (hh[1] * sm).clamp(0.0, 1.0);
            if let Some(t) = hvl {
                hh[2] = (hh[2] + (sample(t, h0) - 0.5) * 0.5).clamp(0.0, 1.0);
            }
            v = hsl_to_rgb(hh[0], hh[1], hh[2]);
        }
        dec(v)
    }

    /// What `fx.wgsl` reads (an `Rgba32Float` texture [`CURVE_N`] wide, 4 rows, flattened): row 0
    /// the scalars below, rows 1–3 the curve tables, four to a row (one per channel).
    ///
    /// Row 0, as texels: gains + exposure · b0 w0 shadows highlights · contrast faded creative sat ·
    /// shadow tint + vibrance · highlight tint + vignette · midpoint roundness feather look ·
    /// look strength, wheels, shadows and midtones lightness · shadow wheel + highlights lightness ·
    /// midtone wheel + advanced · highlight wheel + curve mask · HSL on, hue, range, sat min ·
    /// luma min, luma max, soft, show mask · temp, tint, sat, shift.
    pub fn gpu_data(&self) -> Vec<f32> {
        let mut out = vec![0f32; CURVE_N * 4 * 4];
        let flag = |v: bool| if v { 1.0 } else { 0.0 };
        let mask = self.curves.iter().enumerate().filter(|(_, c)| c.is_some()).map(|(i, _)| 1u32 << i).sum::<u32>();
        let hsl = self.hsl.clone().unwrap_or(LumetriHsl {
            hue: 0.0,
            range: 1.0,
            sat_min: 0.0,
            luma_min: 0.0,
            luma_max: 1.0,
            soft: 1.0,
            show_mask: 0,
            temp: 0.0,
            tint: 0.0,
            sat: 1.0,
            shift: 0.0,
        });
        let [g, st, ht] = [self.gains, self.shadow_tint, self.highlight_tint];
        let [ws, wm, wh] = self.wheel;
        let [ls, lm, lh] = self.lightness;
        let row0 = [
            g[0],
            g[1],
            g[2],
            self.exposure, //
            self.b0,
            self.w0,
            self.shadows,
            self.highlights, //
            self.contrast,
            self.faded,
            flag(self.creative),
            self.sat, //
            st[0],
            st[1],
            st[2],
            self.vibrance, //
            ht[0],
            ht[1],
            ht[2],
            self.vignette, //
            self.v_mid,
            self.v_round,
            self.v_feather,
            self.look as f32, //
            self.look_k,
            flag(self.wheels),
            ls,
            lm, //
            ws[0],
            ws[1],
            ws[2],
            lh, //
            wm[0],
            wm[1],
            wm[2],
            flag(self.advanced), //
            wh[0],
            wh[1],
            wh[2],
            mask as f32, //
            flag(self.hsl.is_some()),
            hsl.hue,
            hsl.range,
            hsl.sat_min, //
            hsl.luma_min,
            hsl.luma_max,
            hsl.soft,
            hsl.show_mask as f32, //
            hsl.temp,
            hsl.tint,
            hsl.sat,
            hsl.shift,
        ];
        for (o, v) in out.iter_mut().zip(row0) {
            *o = v;
        }
        for (i, table) in self.curves.iter().enumerate() {
            if let Some(t) = table {
                let (row, channel) = (1 + i / 4, i % 4);
                for (x, v) in t.iter().take(CURVE_N).enumerate() {
                    if let Some(slot) = out.get_mut((row * CURVE_N + x) * 4 + channel) {
                        *slot = *v;
                    }
                }
            }
        }
        out
    }
}

/// HSL Secondary on one pixel (SDR: the key and the correction on the display-encoded colour).
fn secondary(s: &LumetriHsl, c: [f32; 3]) -> [f32; 3] {
    let u = enc(c).map(|q| q.clamp(0.0, 1.0));
    let m = hsl_key(u, s.hue, s.range, s.sat_min, s.luma_min, s.luma_max, s.soft);
    let out = match s.show_mask {
        1 => {
            let g = rgb_to_hsl(u[0], u[1], u[2])[2];
            [g + (u[0] - g) * m, g + (u[1] - g) * m, g + (u[2] - g) * m]
        }
        2 => u.map(|q| q * m),
        3 => [m, m, m],
        _ => {
            let mut hh = rgb_to_hsl(u[0], u[1], u[2]);
            hh[0] = (hh[0] + s.shift).rem_euclid(1.0);
            hh[1] = (hh[1] * s.sat).clamp(0.0, 1.0);
            let mut c2 = hsl_to_rgb(hh[0], hh[1], hh[2]);
            c2[0] *= 1.0 + 0.25 * s.temp;
            c2[2] *= 1.0 - 0.25 * s.temp;
            c2[1] *= 1.0 - 0.2 * s.tint;
            [u[0] + (c2[0] - u[0]) * m, u[1] + (c2[1] - u[1]) * m, u[2] + (c2[2] - u[2]) * m]
        }
    };
    dec(out)
}

/// `lumetri_advanced`'s table lookup (SDR: no extension above 1).
fn sample(t: &[f32], x: f32) -> f32 {
    let Some(last) = t.len().checked_sub(1) else { return x };
    let p = x.clamp(0.0, 1.0) * last as f32;
    let i = (p as usize).min(last);
    let j = (i + 1).min(last);
    let (a, b) = (t.get(i).copied().unwrap_or(0.0), t.get(j).copied().unwrap_or(0.0));
    a + (b - a) * (p - i as f32)
}

/// `GradeSpace::Sdr` encode / decode.
fn enc(c: [f32; 3]) -> [f32; 3] {
    c.map(|v| linear_to_srgb(v.max(0.0)))
}
fn dec(v: [f32; 3]) -> [f32; 3] {
    v.map(|q| srgb_to_linear(q.clamp(0.0, 1.0)))
}
fn luma(v: [f32; 3]) -> f32 {
    0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_geom::Vec2;
    use filmcraft_project::{ParamValue, find_effect};
    use filmcraft_time::Tick;

    fn cx() -> FxCtx<'static> {
        FxCtx { t: Tick(0), px_scale: 1.0, seconds: 0.0, timecode: "", clip_name: "", project: None, env: None, working: filmcraft_color::WorkingSpace::Rec709 }
    }

    fn lumetri(params: &[(&str, ParamValue)]) -> EffectInstance {
        let mut e = find_effect("lumetri").unwrap().instance();
        for (k, v) in params {
            e.params.get_mut(*k).unwrap_or_else(|| panic!("{k}")).value = v.clone();
        }
        e
    }

    /// A picture with hues, greys, lights and darks and partial alpha.
    fn picture() -> Image {
        let (w, h) = (48, 27);
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let fx = x as f32 / (w - 1) as f32;
                let fy = y as f32 / (h - 1) as f32;
                let c = if x % 7 == 0 { [fy; 3] } else { hsl_to_rgb(fx, 0.2 + 0.8 * ((y % 4) as f32 / 3.0), 0.05 + 0.9 * fy).map(srgb_to_linear) };
                let a = if y % 9 == 0 { 0.4 } else { 1.0 };
                let i = (y * w + x) * 4;
                img.px[i..i + 4].copy_from_slice(&[c[0] * a, c[1] * a, c[2] * a, a]);
            }
        }
        img
    }

    pub(crate) fn grades() -> Vec<Vec<(&'static str, ParamValue)>> {
        let fl = ParamValue::Float;
        let col = |r, g, b| ParamValue::Color([r, g, b, 1.0]);
        let curve = |p: &[[f32; 2]]| ParamValue::Curve(p.to_vec());
        vec![
            vec![],
            vec![("exposure", fl(0.7)), ("contrast", fl(35.0)), ("highlights", fl(-40.0)), ("shadows", fl(30.0)), ("whites", fl(20.0)), ("blacks", fl(-25.0))],
            vec![("temperature", fl(40.0)), ("tint", fl(-30.0)), ("saturation", fl(140.0)), ("contrast", fl(-50.0))],
            vec![
                ("faded_film", fl(40.0)),
                ("vibrance", fl(60.0)),
                ("creative_sat", fl(80.0)),
                ("shadow_tint", col(0.3, 0.5, 0.7)),
                ("highlight_tint", col(0.7, 0.55, 0.4)),
            ],
            vec![("look", ParamValue::Choice(1)), ("look_intensity", fl(80.0))],
            vec![("look", ParamValue::Choice(6)), ("look_intensity", fl(150.0)), ("vignette_amount", fl(-2.0)), ("vignette_roundness", fl(-50.0))],
            vec![("vignette_amount", fl(1.5)), ("vignette_midpoint", fl(30.0)), ("vignette_feather", fl(80.0))],
            vec![
                ("wheel_shadows", ParamValue::Vec2(Vec2::new(0.3, -0.2))),
                ("wheel_midtones", ParamValue::Vec2(Vec2::new(-0.1, 0.4))),
                ("wheel_highlights", ParamValue::Vec2(Vec2::new(0.2, 0.1))),
                ("wheel_midtones_l", fl(-20.0)),
                ("wheel_highlights_l", fl(30.0)),
            ],
            vec![
                ("curve_luma", curve(&[[0.0, 0.0], [0.3, 0.22], [0.7, 0.8], [1.0, 1.0]])),
                ("curve_red", curve(&[[0.0, 0.05], [0.5, 0.55], [1.0, 1.0]])),
                ("curve_blue", curve(&[[0.0, 0.0], [0.6, 0.5], [1.0, 0.95]])),
            ],
            vec![
                ("hue_vs_sat", curve(&[[0.0, 0.5], [0.33, 0.9], [0.66, 0.5]])),
                ("hue_vs_hue", curve(&[[0.1, 0.5], [0.2, 0.6], [0.3, 0.5]])),
                ("hue_vs_luma", curve(&[[0.5, 0.5], [0.6, 0.3], [0.7, 0.5]])),
                ("luma_vs_sat", curve(&[[0.0, 0.3], [0.5, 0.5], [1.0, 0.7]])),
                ("sat_vs_sat", curve(&[[0.2, 0.5], [0.8, 0.8]])),
            ],
            vec![
                ("hsl_on", ParamValue::Bool(true)),
                ("hsl_hue", fl(120.0)),
                ("hsl_hue_range", fl(60.0)),
                ("hsl_sat", fl(30.0)),
                ("hsl_hue_shift", fl(40.0)),
                ("hsl_temp", fl(50.0)),
            ],
            vec![("hsl_on", ParamValue::Bool(true)), ("hsl_hue", fl(20.0)), ("hsl_show_mask", ParamValue::Choice(1))],
            vec![("hsl_on", ParamValue::Bool(true)), ("hsl_show_mask", ParamValue::Choice(3)), ("hsl_soft", fl(0.0))],
            vec![("basic_on", ParamValue::Bool(false)), ("exposure", fl(3.0)), ("creative_on", ParamValue::Bool(false)), ("look", ParamValue::Choice(2))],
            vec![
                ("curves_on", ParamValue::Bool(false)),
                ("curve_luma", curve(&[[0.0, 1.0], [1.0, 0.0]])),
                ("wheels_on", ParamValue::Bool(false)),
                ("wheel_midtones_l", fl(50.0)),
            ],
        ]
    }

    /// The op reproduces `effects::lumetri` (the CPU's own Lumetri) wherever it claims to.
    #[test]
    fn matches_the_cpu_lumetri() {
        for (n, grade) in grades().into_iter().enumerate() {
            let e = lumetri(&grade);
            let op = LumetriOp::eval(&e, &cx()).unwrap_or_else(|| panic!("grade {n} not covered"));
            let mut ours = picture();
            op.apply(&mut ours);
            let mut theirs = picture();
            crate::effects::lumetri(&mut theirs, &e, &cx());
            let worst = ours.px.iter().zip(&theirs.px).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst < 1e-5, "grade {n}: differs by {worst}");
        }
    }

    #[test]
    fn hands_the_rest_to_the_cpu() {
        let fl = ParamValue::Float;
        for grade in [
            vec![("input_lut", ParamValue::Text("builtin:rec709".into()))],
            vec![("look_lut", ParamValue::Text("lib:1".into()))],
            vec![("sharpen", fl(40.0))],
            vec![("hsl_on", ParamValue::Bool(true)), ("hsl_blur", fl(10.0))],
            vec![("hsl_on", ParamValue::Bool(true)), ("hsl_denoise", fl(10.0))],
        ] {
            assert!(LumetriOp::eval(&lumetri(&grade), &cx()).is_none(), "{grade:?}");
        }
        // the switches turn the unsupported parts off
        let off = lumetri(&[("sharpen", fl(40.0)), ("creative_on", ParamValue::Bool(false)), ("hsl_blur", fl(10.0))]);
        assert!(LumetriOp::eval(&off, &cx()).is_some());
    }

    #[test]
    fn gpu_data_layout() {
        let e = lumetri(&[("exposure", ParamValue::Float(1.0)), ("curve_red", ParamValue::Curve(vec![[0.0, 0.1], [1.0, 1.0]]))]);
        let op = LumetriOp::eval(&e, &cx()).unwrap();
        let d = op.gpu_data();
        assert_eq!(d.len(), CURVE_N * 16);
        assert_eq!(d[3], 2.0, "exposure");
        assert_eq!(d[39], 2.0, "curve mask: red only");
        assert!((d[(CURVE_N) * 4 + 1] - 0.1).abs() < 1e-6, "red table, first entry, green channel of row 1");
    }
}
