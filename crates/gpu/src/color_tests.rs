//! GPU-vs-CPU parity of colour-managed layers: log / HDR / wide-gamut Y'CbCr frames converted by
//! their input transform in the compositor (`PlanLayer::color`) against the CPU plan executor.

use super::blend_tests::{device, stats8, yuv_frame};
use super::*;
use filmcraft_color::{ColorInfo, ColorPipeline, ColorSpace, Matrix, Primaries, Range, Transfer};
use filmcraft_geom::Affine;
use filmcraft_render::plan::{FramePlan, PlanLayer, execute_cpu};

/// Gradients in luma and both chroma channels, 4:4:4 so that chroma isn't interpolated (the CPU
/// takes the nearest 4:2:0 sample, the GPU interpolates; HDR gain magnifies that difference,
/// which is a sampling question, not the conversion's).
fn frame(transfer: Transfer) -> Arc<VideoFrame> {
    let (w, h) = (96u32, 54u32);
    let y: Vec<u8> = (0..w * h).map(|i| (16 + ((i % w) * 219 / w)) as u8).collect();
    let u: Vec<u8> = (0..w * h).map(|i| (64 + (i / w) * 128 / h) as u8).collect();
    let v: Vec<u8> = (0..w * h).map(|i| (200 - (i % w) * 100 / w) as u8).collect();
    Arc::new(VideoFrame {
        width: w,
        height: h,
        data: PixelData::Yuv8 { planes: [Arc::new(y), Arc::new(u), Arc::new(v)], chroma: filmcraft_frame::Chroma::C444, alpha: None },
        color: ColorInfo { matrix: Matrix::Bt2020Ncl, transfer, primaries: Primaries::Bt2020, range: Range::Limited },
        par: (1, 1),
        pts: Default::default(),
    })
}

#[test]
fn managed_layers_match_cpu() {
    let Some((dev, q)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut c = GpuCompositor::new(&dev, &q);
    let cases = [
        (ColorSpace::Rec2100Hlg, Transfer::Hlg),
        (ColorSpace::Rec2100Pq, Transfer::Pq),
        (ColorSpace::AppleLog, Transfer::Bt709),
        (ColorSpace::SLog3SGamut3Cine, Transfer::Bt709),
        (ColorSpace::Rec2020, Transfer::Bt709),
    ];
    for (cs, transfer) in cases {
        let t = filmcraft_render::colorman::input_transform(cs, Range::Limited, &ColorPipeline::REC709, None);
        assert!(t.gpu_ready(), "{cs:?}");
        // drawn 1:1, and scaled down (supersampled: the stage runs on the average, as on the CPU)
        for matrix in [Affine::IDENTITY, Affine::scale(0.5, 0.5)] {
            let plan = FramePlan::Layers {
                width: 96,
                height: 54,
                layers: vec![PlanLayer { color: Some(t.clone()), ..PlanLayer::new(frame(transfer), matrix, 1.0, Blend::Normal) }],
            };
            let cpu_img = execute_cpu(&plan);
            let cpu = cpu_img.over_black_rgba8();
            c.composite(&plan);
            let (_, _, gpu) = c.read_output().expect("readback");
            // inside the picture (the half-size layer covers the top-left quarter)
            let keep: Vec<bool> = (0..96 * 54).map(|i| matrix == Affine::IDENTITY || ((i % 96) < 46 && (i / 96) < 25)).collect();
            let (p99, mean, max) = stats8(&cpu, &gpu, &keep);
            eprintln!("{cs:?} {:?}: 8-bit p99 {p99} mean {mean:.3} max {max}", matrix.a);
            if matrix == Affine::IDENTITY {
                assert!(p99 <= 2 && mean < 0.5, "{cs:?}: p99 {p99}, mean {mean}, max {max}");
            } else {
                // shrunk: the GPU averages the footprint and then tone-maps (as the CPU renderer
                // does when it decodes at a reduced size); this reference converts every pixel
                // and then shrinks, so steep gradients differ a little
                assert!(p99 <= 8 && mean < 1.0, "{cs:?} shrunk: p99 {p99}, mean {mean}, max {max}");
            }
        }
    }
}

#[test]
fn plain_layers_skip_the_table() {
    // an unmanaged layer next to a managed one: each keeps its own conversion
    let Some((dev, q)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut c = GpuCompositor::new(&dev, &q);
    let t = filmcraft_render::colorman::input_transform(ColorSpace::Rec2100Hlg, Range::Limited, &ColorPipeline::REC709, None);
    let plan = FramePlan::Layers {
        width: 96,
        height: 54,
        layers: vec![
            PlanLayer::new(yuv_frame(96, 54), Affine::IDENTITY, 1.0, Blend::Normal),
            PlanLayer { color: Some(t), ..PlanLayer::new(frame(Transfer::Hlg), Affine::translate(48.0, 0.0), 1.0, Blend::Normal) },
        ],
    };
    let cpu = execute_cpu(&plan).over_black_rgba8();
    c.composite(&plan);
    let (_, _, gpu) = c.read_output().expect("readback");
    let keep: Vec<bool> = (0..96 * 54).map(|i| (i % 96) != 47 && (i % 96) != 48).collect();
    // the plain half is 4:2:0 (chroma interpolated on the GPU, nearest on the CPU, as for any
    // SDR layer); the managed half must match exactly
    let left: Vec<bool> = keep.iter().enumerate().map(|(i, k)| *k && (i % 96) < 47).collect();
    let right: Vec<bool> = keep.iter().enumerate().map(|(i, k)| *k && (i % 96) > 48).collect();
    let (lp99, lmean, _) = stats8(&cpu, &gpu, &left);
    let (rp99, rmean, _) = stats8(&cpu, &gpu, &right);
    eprintln!("plain p99 {lp99} mean {lmean:.3}; managed p99 {rp99} mean {rmean:.3}");
    assert!(lp99 <= 6 && lmean < 2.5, "plain: p99 {lp99}, mean {lmean}");
    assert!(rp99 <= 2 && rmean < 0.5, "managed: p99 {rp99}, mean {rmean}");
}

/// Flat colours (no resampling, no chroma interpolation): the conversion alone.
#[test]
fn managed_flat_colours_match_cpu() {
    let Some((dev, q)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut c = GpuCompositor::new(&dev, &q);
    for (cs, transfer) in [(ColorSpace::Rec2100Hlg, Transfer::Hlg), (ColorSpace::Rec2100Pq, Transfer::Pq), (ColorSpace::AppleLog, Transfer::Bt709)] {
        let t = filmcraft_render::colorman::input_transform(cs, Range::Limited, &ColorPipeline::REC709, None);
        for (y, u, v) in [(64u8, 128u8, 128u8), (180, 128, 128), (235, 128, 128), (120, 90, 170), (200, 150, 100), (40, 140, 120)] {
            let (w, h) = (8u32, 8u32);
            let f = VideoFrame {
                width: w,
                height: h,
                data: PixelData::Yuv8 {
                    planes: [Arc::new(vec![y; 64]), Arc::new(vec![u; 16]), Arc::new(vec![v; 16])],
                    chroma: filmcraft_frame::Chroma::C420,
                    alpha: None,
                },
                color: ColorInfo { matrix: Matrix::Bt2020Ncl, transfer, primaries: Primaries::Bt2020, range: Range::Limited },
                par: (1, 1),
                pts: Default::default(),
            };
            let plan = FramePlan::Layers {
                width: 8,
                height: 8,
                layers: vec![PlanLayer { color: Some(t.clone()), ..PlanLayer::new(Arc::new(f), Affine::IDENTITY, 1.0, Blend::Normal) }],
            };
            let cpu = execute_cpu(&plan);
            c.composite(&plan);
            let gpu = super::blend_tests::read_accum(&c);
            let (a, b) = (&cpu.px[..4], &gpu[..4]);
            eprintln!("{cs:?} yuv ({y},{u},{v}): cpu {a:?} gpu {b:?}");
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() <= 2e-3 * (1.0 + a[k].abs()), "{cs:?} ({y},{u},{v}) channel {k}: cpu {} gpu {}", a[k], b[k]);
            }
        }
    }
}
