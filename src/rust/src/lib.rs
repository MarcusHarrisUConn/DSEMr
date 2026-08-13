use extendr_api::prelude::*;

/// Compute Gaussian regression sufficient statistics without calling the R API
/// from worker threads.
///
/// @param x Column-major numeric design matrix.
/// @param y Numeric response vector.
/// @param n Number of rows.
/// @param p Number of columns.
/// @return Internal sufficient-statistics list.
/// @keywords internal
#[extendr]
fn dsem_ar_sufficient(
    x: Vec<f64>,
    y: Vec<f64>,
    n: i32,
    p: i32,
) -> std::result::Result<List, Error> {
    if n <= 0 || p <= 0 {
        return Err(Error::Other("n and p must be positive".into()));
    }
    let n = n as usize;
    let p = p as usize;
    let (xtx, xty, yty) =
        sufficient_kernel(&x, &y, n, p).map_err(|message| Error::Other(message.into()))?;

    Ok(list!(
        xtx = xtx,
        xty = xty,
        yty = yty,
        n = n as i32,
        p = p as i32
    ))
}

fn sufficient_kernel(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
) -> std::result::Result<(Vec<f64>, Vec<f64>, f64), &'static str> {
    if x.len() != n * p {
        return Err("x length does not match n * p");
    }
    if y.len() != n {
        return Err("y length does not match n");
    }
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    let mut yty = 0.0;
    for row in 0..n {
        let response = y[row];
        yty += response * response;
        for col in 0..p {
            let x_col = x[row + n * col];
            xty[col] += x_col * response;
            for other in 0..p {
                xtx[col + p * other] += x_col * x[row + n * other];
            }
        }
    }
    Ok((xtx, xty, yty))
}

extendr_module! {
    mod dsemr;
    fn dsem_ar_sufficient;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_cross_products() {
        // Matrix columns are [1, 1, 1] and [1, 2, 3].
        let (xtx, xty, yty) =
            sufficient_kernel(&[1.0, 1.0, 1.0, 1.0, 2.0, 3.0], &[2.0, 4.0, 6.0], 3, 2).unwrap();
        assert_eq!(xtx, vec![3.0, 6.0, 6.0, 14.0]);
        assert_eq!(xty, vec![12.0, 28.0]);
        assert_eq!(yty, 56.0);
    }
}
