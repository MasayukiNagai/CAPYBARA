import numpy as np
import scipy.special
import scipy.ndimage


def multinomial_log_probs(
    category_log_probs, trials, query_counts, return_cross_entropy=True
):
    log_n_fact = scipy.special.gammaln(trials + 1)
    log_counts_fact = scipy.special.gammaln(query_counts + 1)
    log_counts_fact_sum = np.sum(log_counts_fact, axis=-1)
    log_prob_pows = category_log_probs * query_counts
    log_prob_pows_sum = np.sum(log_prob_pows, axis=-1)

    log_prob = log_n_fact - log_counts_fact_sum + log_prob_pows_sum
    if return_cross_entropy:
        cross_ent = (-log_prob_pows_sum) / trials
        return log_prob, cross_ent
    else:
        return log_prob


def profile_multinomial_nll(
    true_profs, log_pred_profs, true_counts, prof_smooth_kernel_sigma,
    prof_smooth_kernel_width, smooth_pred_profs=False,
    return_cross_entropy=True, batch_size=200
):
    num_samples = true_profs.shape[0]
    num_tasks = true_profs.shape[1]
    nlls = np.empty((num_samples, num_tasks))
    if return_cross_entropy:
        ces = np.empty((num_samples, num_tasks))

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        true_profs_batch = true_profs[start:end]
        log_pred_profs_batch = log_pred_profs[start:end]
        true_counts_batch = true_counts[start:end]

        true_profs_batch = np.swapaxes(true_profs_batch, 2, 3)
        log_pred_profs_batch = np.swapaxes(log_pred_profs_batch, 2, 3)

        if prof_smooth_kernel_width == 0:
            sigma, truncate = 1, 0
        else:
            sigma = prof_smooth_kernel_sigma
            truncate = (prof_smooth_kernel_width - 1) / (2 * sigma)
        if smooth_pred_profs:
            pred_profs_batch = np.exp(log_pred_profs_batch)
            pred_profs_batch_smooth = scipy.ndimage.gaussian_filter1d(
                pred_profs_batch, sigma, axis=-1, truncate=truncate
            )
            log_pred_profs_batch = np.log(pred_profs_batch_smooth)

        result_batch = multinomial_log_probs(
            log_pred_profs_batch, true_counts_batch, true_profs_batch,
            return_cross_entropy
        )
        if return_cross_entropy:
            nll_batch, ce_batch = result_batch
            ce_batch_mean = np.mean(ce_batch, axis=2)
            ces[start:end] = ce_batch_mean
        else:
            nll_batch = result_batch
        nll_batch_mean = np.mean(-nll_batch, axis=2)
        nlls[start:end] = nll_batch_mean

    if return_cross_entropy:
        return nlls, ces
    return nlls


def _kl_divergence(probs1, probs2):
    quot = np.divide(
        probs1, probs2, out=np.ones_like(probs1),
        where=((probs1 != 0) & (probs2 != 0))
    )
    return np.sum(probs1 * np.log(quot), axis=-1)


def jensen_shannon_distance(probs1, probs2):
    probs1_sum = np.sum(probs1, axis=-1, keepdims=True)
    probs1 = np.divide(
        probs1, probs1_sum, out=np.full_like(probs1, np.nan),
        where=(probs1_sum != 0)
    )
    probs2_sum = np.sum(probs2, axis=-1, keepdims=True)
    probs2 = np.divide(
        probs2, probs2_sum, out=np.full_like(probs2, np.nan),
        where=(probs2_sum != 0)
    )
    mid = 0.5 * (probs1 + probs2)
    return 0.5 * (_kl_divergence(probs1, mid) + _kl_divergence(probs2, mid))


def profile_jsd(
    true_prof_probs, pred_prof_probs, prof_smooth_kernel_sigma,
    prof_smooth_kernel_width, smooth_true_profs=True, smooth_pred_profs=False,
    batch_size=200
):
    num_samples = true_prof_probs.shape[0]
    num_tasks = true_prof_probs.shape[1]
    jsds = np.empty((num_samples, num_tasks))

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        true_prof_probs_batch = true_prof_probs[start:end]
        pred_prof_probs_batch = pred_prof_probs[start:end]

        if np.min(true_prof_probs_batch) < 0:
            true_prof_probs_batch = np.exp(true_prof_probs_batch)
        if np.min(pred_prof_probs_batch) < 0:
            pred_prof_probs_batch = np.exp(pred_prof_probs_batch)

        true_prof_swap = np.swapaxes(true_prof_probs_batch, 2, 3)
        pred_prof_swap = np.swapaxes(pred_prof_probs_batch, 2, 3)

        if prof_smooth_kernel_width == 0:
            sigma, truncate = 1, 0
        else:
            sigma = prof_smooth_kernel_sigma
            truncate = (prof_smooth_kernel_width - 1) / (2 * sigma)
        if smooth_true_profs:
            true_prof_swap = scipy.ndimage.gaussian_filter1d(
                true_prof_swap, sigma, axis=-1, truncate=truncate
            )
        if smooth_pred_profs:
            pred_prof_swap = scipy.ndimage.gaussian_filter1d(
                pred_prof_swap, sigma, axis=-1, truncate=truncate
            )

        jsd_batch = jensen_shannon_distance(true_prof_swap, pred_prof_swap)
        jsd_batch_mean = np.nanmean(jsd_batch, axis=-1)
        jsds[start:end] = jsd_batch_mean
    return jsds


def pearson_corr(arr1, arr2):
    mean1 = np.mean(arr1, axis=-1, keepdims=True)
    mean2 = np.mean(arr2, axis=-1, keepdims=True)
    dev1, dev2 = arr1 - mean1, arr2 - mean2
    numer = np.sum(dev1 * dev2, axis=-1)
    var1, var2 = np.sum(np.square(dev1), axis=-1), np.sum(np.square(dev2), axis=-1)
    denom = np.sqrt(var1 * var2)
    return np.divide(numer, denom, out=np.full_like(numer, np.nan), where=(denom != 0))


def average_ranks(arr):
    sorted_inds = np.argsort(arr, axis=-1)
    ranks, ranges = np.empty_like(arr), np.empty_like(arr)
    ranges = np.tile(np.arange(arr.shape[-1]), arr.shape[:-1] + (1,))
    np.put_along_axis(ranks, sorted_inds, ranges, -1)
    ranks = ranks.astype(int)

    sorted_arr = np.take_along_axis(arr, sorted_inds, axis=-1)
    diffs = np.diff(sorted_arr, axis=-1)
    del sorted_arr
    pad_diffs = np.pad(diffs, ([(0, 0)] * (diffs.ndim - 1)) + [(1, 0)])
    del diffs
    pad_diffs[pad_diffs != 0] = 1
    unique_inds = np.cumsum(pad_diffs, axis=-1).astype(int)
    del pad_diffs

    unique_maxes = np.zeros_like(arr)
    np.put_along_axis(unique_maxes, unique_inds, ranges, -1)
    diff = np.diff(unique_maxes, prepend=-1, axis=-1)
    unique_avgs = unique_maxes - ((diff - 1) / 2)
    del unique_maxes, diff

    return np.take_along_axis(unique_avgs, np.take_along_axis(unique_inds, ranks, -1), -1)


def spearman_corr(arr1, arr2):
    return pearson_corr(average_ranks(arr1), average_ranks(arr2))


def mean_squared_error(arr1, arr2):
    return np.mean(np.square(arr1 - arr2), axis=-1)


def r2_score(arr1, arr2):
    ss_res = np.sum(np.square(arr1 - arr2), axis=-1)
    ss_tot = np.sum(np.square(arr1 - np.mean(arr1, axis=-1, keepdims=True)), axis=-1)
    return 1 - np.divide(ss_res, ss_tot, out=np.full_like(ss_res, np.nan), where=(ss_tot != 0))


def profile_corr_mse(
    true_prof_probs, pred_prof_probs, prof_smooth_kernel_sigma,
    prof_smooth_kernel_width, smooth_true_profs=True, smooth_pred_profs=False,
    batch_size=200
):
    num_samples, num_tasks = true_prof_probs.shape[:2]
    pears = np.zeros((num_samples, num_tasks))
    spear = np.zeros((num_samples, num_tasks))
    mse = np.zeros((num_samples, num_tasks))

    if prof_smooth_kernel_width == 0:
        sigma, truncate = 1, 0
    else:
        sigma = prof_smooth_kernel_sigma
        truncate = (prof_smooth_kernel_width - 1) / (2 * sigma)

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        true_batch = true_prof_probs[start:end]
        pred_batch = pred_prof_probs[start:end]

        if np.min(true_batch) < 0:
            true_batch = np.exp(true_batch)
        if np.min(pred_batch) < 0:
            pred_batch = np.exp(pred_batch)

        true_batch_sum = np.sum(true_batch, axis=2, keepdims=True)
        if np.max(true_batch_sum) > 1.5:
            true_batch = np.divide(
                true_batch, true_batch_sum,
                out=np.zeros_like(true_batch, dtype=float),
                where=(true_batch_sum != 0)
            )
        pred_batch_sum = np.sum(pred_batch, axis=2, keepdims=True)
        if np.max(pred_batch_sum) > 1.5:
            pred_batch = np.divide(
                pred_batch, pred_batch_sum,
                out=np.zeros_like(pred_batch, dtype=float),
                where=(pred_batch_sum != 0)
            )

        if smooth_true_profs:
            true_batch = scipy.ndimage.gaussian_filter1d(true_batch, sigma, axis=2, truncate=truncate)
        if smooth_pred_profs:
            pred_batch = scipy.ndimage.gaussian_filter1d(pred_batch, sigma, axis=2, truncate=truncate)

        new_shape = (true_batch.shape[0], num_tasks, -1)
        true_flat = np.reshape(true_batch, new_shape)
        pred_flat = np.reshape(pred_batch, new_shape)

        pears[start:end] = pearson_corr(true_flat, pred_flat)
        spear[start:end] = spearman_corr(true_flat, pred_flat)
        mse[start:end] = mean_squared_error(true_flat, pred_flat)

    return pears, spear, mse


def count_corr_mse(log_true_total_counts, log_pred_total_counts):
    pears, spear, mse, _ = count_corr_mse_r2(log_true_total_counts, log_pred_total_counts)
    return pears, spear, mse


def count_corr_mse_r2(log_true_total_counts, log_pred_total_counts):
    num_tasks = log_true_total_counts.shape[1]
    log_true_total_counts = np.reshape(
        np.swapaxes(log_true_total_counts, 0, 1), (num_tasks, -1)
    )
    log_pred_total_counts = np.reshape(
        np.swapaxes(log_pred_total_counts, 0, 1), (num_tasks, -1)
    )
    pears = pearson_corr(log_true_total_counts, log_pred_total_counts)
    spear = spearman_corr(log_true_total_counts, log_pred_total_counts)
    mse = mean_squared_error(log_true_total_counts, log_pred_total_counts)
    r2 = r2_score(log_true_total_counts, log_pred_total_counts)
    return pears, spear, mse, r2


def compute_performance_metrics(
    true_profs, log_pred_profs, true_counts, log_pred_counts,
    prof_smooth_kernel_sigma=7, prof_smooth_kernel_width=81, smooth_true_profs=True,
    smooth_pred_profs=False
):
    assert true_profs.shape == log_pred_profs.shape, (true_profs.shape, log_pred_profs.shape)
    assert true_counts.shape == log_pred_counts.shape, (true_counts.shape, log_pred_counts.shape)
    assert len(true_profs.shape) == 4, true_profs.shape
    assert len(true_counts.shape) == 3, true_counts.shape
    assert true_profs.shape[:2] == true_counts.shape[:2], (true_profs.shape, true_counts.shape)
    assert np.all(true_profs % 1 == 0), [n for n in true_profs.flatten() if n % 1 != 0]
    assert np.all(true_counts % 1 == 0), [n for n in true_counts.flatten() if n % 1 != 0]
    assert np.all(log_pred_profs <= 0), [n for n in log_pred_profs.flatten() if n >= 0]

    nll, ce = profile_multinomial_nll(
        true_profs, log_pred_profs, true_counts, prof_smooth_kernel_sigma,
        prof_smooth_kernel_width, smooth_pred_profs=smooth_pred_profs,
        return_cross_entropy=True
    )
    jsd = profile_jsd(
        true_profs, log_pred_profs, prof_smooth_kernel_sigma,
        prof_smooth_kernel_width, smooth_true_profs=smooth_true_profs,
        smooth_pred_profs=smooth_pred_profs
    )
    prof_pears, prof_spear, prof_mse = profile_corr_mse(
        true_profs, log_pred_profs, prof_smooth_kernel_sigma,
        prof_smooth_kernel_width, smooth_true_profs=smooth_true_profs,
        smooth_pred_profs=smooth_pred_profs
    )
    log_true_counts = np.log(true_counts + 1)
    count_pears, count_spear, count_mse, count_r2 = count_corr_mse_r2(
        log_true_counts, log_pred_counts
    )

    return {
        "nll": nll,
        "cross_ent": ce,
        "jsd": jsd,
        "profile_pearson": prof_pears,
        "profile_spearman": prof_spear,
        "profile_mse": prof_mse,
        "count_pearson": count_pears,
        "count_spearman": count_spear,
        "count_mse": count_mse,
        "count_r2": count_r2,
    }
