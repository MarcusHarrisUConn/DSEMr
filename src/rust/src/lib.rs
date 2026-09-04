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

/// Compute sufficient statistics for contiguous groups in a Gaussian model.
///
/// @param x Column-major numeric design matrix.
/// @param y Numeric response vector.
/// @param group_sizes Number of consecutive rows in each group.
/// @param n Total number of rows.
/// @param p Number of columns.
/// @return Concatenated group sufficient statistics.
/// @keywords internal
#[extendr]
fn dsem_grouped_sufficient(
    x: Vec<f64>,
    y: Vec<f64>,
    group_sizes: Vec<i32>,
    n: i32,
    p: i32,
) -> std::result::Result<List, Error> {
    if n <= 0 || p <= 0 || group_sizes.is_empty() {
        return Err(Error::Other(
            "n, p, and group_sizes must be positive".into(),
        ));
    }
    let n_usize = n as usize;
    let p_usize = p as usize;
    let sizes: std::result::Result<Vec<usize>, _> = group_sizes
        .iter()
        .map(|size| {
            if *size <= 0 {
                Err("every group size must be positive")
            } else {
                Ok(*size as usize)
            }
        })
        .collect();
    let sizes = sizes.map_err(|message| Error::Other(message.into()))?;
    let (xtx, xty, yty) = grouped_sufficient_kernel(&x, &y, &sizes, n_usize, p_usize)
        .map_err(|message| Error::Other(message.into()))?;
    Ok(list!(
        xtx = xtx,
        xty = xty,
        yty = yty,
        group_sizes = group_sizes,
        groups = sizes.len() as i32,
        p = p
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

type GroupedStatistics = (Vec<f64>, Vec<f64>, Vec<f64>);

fn grouped_sufficient_kernel(
    x: &[f64],
    y: &[f64],
    group_sizes: &[usize],
    n: usize,
    p: usize,
) -> std::result::Result<GroupedStatistics, &'static str> {
    if x.len() != n * p || y.len() != n {
        return Err("x or y dimensions do not match n and p");
    }
    if group_sizes.iter().sum::<usize>() != n {
        return Err("group sizes do not sum to n");
    }
    let groups = group_sizes.len();
    let mut xtx = vec![0.0; groups * p * p];
    let mut xty = vec![0.0; groups * p];
    let mut yty = vec![0.0; groups];
    let mut start = 0usize;
    for (group, size) in group_sizes.iter().enumerate() {
        for row in start..(start + size) {
            let response = y[row];
            yty[group] += response * response;
            for col in 0..p {
                let x_col = x[row + n * col];
                xty[group * p + col] += x_col * response;
                for other in 0..p {
                    xtx[group * p * p + col + p * other] += x_col * x[row + n * other];
                }
            }
        }
        start += size;
    }
    Ok((xtx, xty, yty))
}

extendr_module! {
    mod dsemr;
    fn dsem_ar_sufficient;
    fn dsem_grouped_sufficient;
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

    #[test]
    fn computes_grouped_cross_products() {
        let (xtx, xty, yty) = grouped_sufficient_kernel(
            &[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0],
            &[2.0, 4.0, 6.0, 8.0],
            &[2, 2],
            4,
            2,
        )
        .unwrap();
        assert_eq!(xtx, vec![2.0, 3.0, 3.0, 5.0, 2.0, 7.0, 7.0, 25.0]);
        assert_eq!(xty, vec![6.0, 10.0, 14.0, 50.0]);
        assert_eq!(yty, vec![20.0, 100.0]);
    }
}
