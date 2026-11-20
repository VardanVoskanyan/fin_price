use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Exp, Normal, Poisson, Uniform};
use rayon::prelude::*;

/// Generator for one path: ChaCha8 keyed by `seed`, on stream `path`.
///
/// Each path gets its own stream, so the result depends only on (seed, path index) and
/// not on how rayon schedules the paths over threads.
fn path_rng(seed: u64, path: usize) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(path as u64);
    rng
}

// ---------------------------------------------------------------------------
// Per-path simulation helpers
// ---------------------------------------------------------------------------

fn simulate_path_merton(
    rng: &mut impl Rng,
    t1: f64,
    t2: f64,
    lmbd: f64,
    k2: f64,
    m: f64,
    delta: f64,
) -> (f64, f64) {
    let intensity = lmbd * (t2 - t1);
    let n = Poisson::new(intensity).unwrap().sample(rng) as usize;

    if n == 0 {
        return (0.0, 0.0);
    }

    let time_dist = Uniform::new(t1, t2);
    let size_dist = Normal::new(m, delta).unwrap();

    let mut total = 0.0_f64;
    let mut exp_weighted = 0.0_f64;

    for _ in 0..n {
        let tau = time_dist.sample(rng);
        let y = size_dist.sample(rng);
        total += y;
        exp_weighted += y * (k2 * tau).exp();
    }

    (total, exp_weighted)
}

fn simulate_path_kou(
    rng: &mut impl Rng,
    t1: f64,
    t2: f64,
    lmbd: f64,
    k2: f64,
    p: f64,
    alpha_plus: f64,
    alpha_minus: f64,
) -> (f64, f64) {
    let intensity = lmbd * (t2 - t1);
    let n = Poisson::new(intensity).unwrap().sample(rng) as usize;

    if n == 0 {
        return (0.0, 0.0);
    }

    let time_dist = Uniform::new(t1, t2);
    let dir_dist = Uniform::new(0.0_f64, 1.0_f64);
    let up_dist = Exp::new(alpha_plus).unwrap();
    let down_dist = Exp::new(alpha_minus).unwrap();

    let mut total = 0.0_f64;
    let mut exp_weighted = 0.0_f64;

    for _ in 0..n {
        let tau = time_dist.sample(rng);
        let y = if dir_dist.sample(rng) < p {
            up_dist.sample(rng)       // positive jump
        } else {
            -down_dist.sample(rng)    // negative jump
        };
        total += y;
        exp_weighted += y * (k2 * tau).exp();
    }

    (total, exp_weighted)
}

// ---------------------------------------------------------------------------
// Python-exposed functions
// ---------------------------------------------------------------------------

/// Simulate jump totals for the Merton (Gaussian) model.
///
/// Each path independently draws a Poisson number of jumps in [t1, t2],
/// then accumulates the sum and exp-weighted sum of Gaussian jump sizes.
/// Paths are processed in parallel via rayon; `seed` makes the result reproducible.
///
/// Returns (total_jumps, exp_weighted_jumps) as 1-D numpy float64 arrays.
#[pyfunction]
fn simul_total_jumps_merton<'py>(
    py: Python<'py>,
    num_paths: usize,
    t1: f64,
    t2: f64,
    lmbd: f64,
    k2: f64,
    m: f64,
    delta: f64,
    seed: u64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let results: Vec<(f64, f64)> = (0..num_paths)
        .into_par_iter()
        .map(|i| {
            let mut rng = path_rng(seed, i);
            simulate_path_merton(&mut rng, t1, t2, lmbd, k2, m, delta)
        })
        .collect();

    unzip_to_arrays(py, results)
}

/// Simulate jump totals for the Kou (double-exponential) model.
///
/// Each path independently draws a Poisson number of jumps in [t1, t2],
/// then accumulates the sum and exp-weighted sum of Kou jump sizes.
/// Paths are processed in parallel via rayon; `seed` makes the result reproducible.
///
/// Returns (total_jumps, exp_weighted_jumps) as 1-D numpy float64 arrays.
#[pyfunction]
fn simul_total_jumps_kou<'py>(
    py: Python<'py>,
    num_paths: usize,
    t1: f64,
    t2: f64,
    lmbd: f64,
    k2: f64,
    p: f64,
    alpha_plus: f64,
    alpha_minus: f64,
    seed: u64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let results: Vec<(f64, f64)> = (0..num_paths)
        .into_par_iter()
        .map(|i| {
            let mut rng = path_rng(seed, i);
            simulate_path_kou(&mut rng, t1, t2, lmbd, k2, p, alpha_plus, alpha_minus)
        })
        .collect();

    unzip_to_arrays(py, results)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

fn unzip_to_arrays<'py>(
    py: Python<'py>,
    results: Vec<(f64, f64)>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let total: Vec<f64> = results.iter().map(|r| r.0).collect();
    let exp_w: Vec<f64> = results.iter().map(|r| r.1).collect();
    (total.into_pyarray_bound(py), exp_w.into_pyarray_bound(py))
}

#[pymodule]
fn _sim(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simul_total_jumps_merton, m)?)?;
    m.add_function(wrap_pyfunction!(simul_total_jumps_kou, m)?)?;
    Ok(())
}
