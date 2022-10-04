import numpy as np
from scipy import stats


class CPMMasker:
    def __init__(self, x_, y_, corrs_=None, ts_=None, fs_=None, binary=False, categorical=False):

        self.mask_cache = {}
        self.corrs = corrs_
        self.ts = ts_
        self.dofs = None
        self.fs = fs_

        if binary:
            if self.ts is None:
                categories = np.unique(y_)
                if len(categories) != 2:
                    raise ValueError("Binary classification requires two classes")
                y_ = np.array([categories.tolist().index(y) for y in y_])
                means = np.array([np.mean(x_[y_ == i], axis=0) for i in range(2)])
                sds = np.array([np.std(x_[y_ == i], axis=0) for i in range(2)])
                vars = sds ** 2
                ns = np.array([np.sum(y_ == i) for i in range(2)])

                self.ts = (means[1] - means[0]) / np.sqrt(vars[1] / ns[1] + vars[0] / ns[0])
                self.dofs = (vars[1] / ns[1] + vars[0] / ns[0]) ** 2 / (
                        (vars[1] / ns[1]) ** 2 / (ns[1] - 1) + (vars[0] / ns[0]) ** 2 / (ns[0] - 1)
                )
        elif categorical:
            if self.fs is None:
                raise NotImplementedError("This has not been tested yet")

                categories = np.unique(y_)
                self.k = categories.shape[0]
                means = np.array([np.mean(x_[y_ == c], axis=0) for c in categories])
                means_one_removed = np.array([np.mean(x_[y_ != c], axis=0) for c in categories])
                sds = np.array([np.std(x_[y_ == c], axis=0) for c in categories])
                sds_one_removed = np.array([np.std(x_[y_ != c], axis=0) for c in categories])
                vars = sds ** 2
                vars_one_removed = sds_one_removed ** 2
                ns = np.array([y_[y_ == c].shape[0] for c in categories])
                ns_one_removed = np.array([y_[y_ != c].shape[0] for c in categories])
                wjs = ns / sds ** 2
                ws = wjs.sum(axis=0)
                x_bars = (wjs * means).sum(axis=0) / ws
                self.fs = (1 / (self.k + 1) * np.sum(wjs * (means - x_bars) ** 2, axis=0)) / (
                        1 + 2 * (self.k - 2) / (self.k ** 2 - 1) * np.sum(1 / (ns - 1) * (1 - wjs / ws) ** 2, axis=0)
                )
                self.f_dofs = (self.k ** 2 - 1) / (3 * np.sum(1 / (ns - 1) * (1 - wjs / ws) ** 2, axis=0))
                self.ts = np.array([(means[i] - means_one_removed[i])/np.sqrt(
                    vars[i]/ns[i] + vars_one_removed[i]/ns_one_removed[i]
                ) for i in range(self.k)])
                self.t_dofs = np.array([(vars[i]/ns[i] + vars_one_removed[i]/ns_one_removed[i])**2 / (
                        (vars[i]/ns[i])**2/(ns[i]-1) + (vars_one_removed[i]/ns_one_removed[i])**2/(ns_one_removed[i]-1)
                ) for i in range(self.k)])
        else:
            if self.corrs is None:
                y_corr = (y_ - y_.mean()) / np.linalg.norm(y_)
                x_corr = (x_ - x_.mean(axis=0)) / np.linalg.norm(x_, axis=0)
                x_corr = np.nan_to_num(x_corr)
                self.corrs = x_corr.T @ y_corr
                self.dofs = np.ones(x_.shape[1]) * (x_.shape[0] - 2)

        self.categorical = categorical
        self.binary = binary
        self.x = x_
        self.y = y_

    def clone(self):
        return CPMMasker(self.x, self.y, self.corrs)

    def critical_fs(self, alpha):
        return stats.f.ppf(1 - alpha / 2, self.k - 1, self.f_dofs)

    def critical_ts(self, alpha):
        return stats.t.ppf(1 - alpha / 2, self.dofs)

    def critical_rs(self, alpha):
        critical_ts = self.critical_ts(alpha)
        return critical_ts / np.sqrt(self.dofs + critical_ts ** 2)

    def get_mask_by_type(self, threshold, mask_type, as_digits=False):
        if mask_type == "positive" or mask_type == "pos" or mask_type == "p":
            return self.get_pos_mask(threshold, as_digits)
        elif mask_type == "negative" or mask_type == "neg" or mask_type == "n":
            return self.get_neg_mask(threshold, as_digits)
        else:
            return self.get_all_mask(threshold, as_digits)

    def get_pos_mask(self, threshold=0.05, as_ones=False):
        cache_key = (threshold, as_ones, "pos")
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        if self.binary:
            critical_ts = self.critical_ts(threshold)
            if as_ones:
                ret = np.ones(self.ts.shape)
                ret[self.ts < critical_ts] = 0
            else:
                ret = self.ts > critical_ts
        elif self.categorical:

            raise NotImplementedError("This has not been tested yet")

            critical_fs = self.critical_fs(threshold)
            if as_ones:
                ret = np.ones((self.fs.shape, self.k))
                non_sig_f_indices = np.where(self.fs < critical_fs)
                ret[non_sig_f_indices, :] = 0 # zero out the parameters for all classes if the F is not significant
                non_sig_t_indices = np.where(self.ts < stats.t.ppf(1 - threshold / 2, self.t_dofs))
                ret[non_sig_t_indices] = 0 # zero out the parameters if the T is not significant
            else:
                ret = np.array(self.fs > critical_fs)
                ret = np.logical_and(ret, self.ts > stats.t.ppf(1 - threshold / 2, self.t_dofs))
        else:
            critical_rs = self.critical_rs(threshold)
            if as_ones:
                ret = np.ones(self.corrs.shape)
                ret[self.corrs < critical_rs] = 0
            else:
                ret = self.corrs > critical_rs

        self.mask_cache[cache_key] = ret
        return ret

    def sort_key(self, i, mask_type):
        if self.binary:
            t = self.ts[i]
            if mask_type == "positive" or mask_type == "pos" or mask_type == "p":
                return t
            elif mask_type == "negative" or mask_type == "neg" or mask_type == "n":
                return -t
            else:
                return abs(t)
        elif self.categorical:
            f = self.fs[i]
            if mask_type == "positive" or mask_type == "pos" or mask_type == "p":
                return f
            elif mask_type == "negative" or mask_type == "neg" or mask_type == "n":
                return -f
            else:
                return abs(f)
        else:
            corr = self.corrs[i]
            if mask_type == "positive":
                return corr
            elif mask_type == "negative":
                return -corr
            else:
                return abs(corr)

    def get_coef_order(self, mask_type):
        if self.binary:
            return sorted(range(self.ts.shape[0]), key=lambda i: self.sort_key(i, mask_type), reverse=True)
        elif self.categorical:
            return sorted(range(self.fs.shape[0]), key=lambda i: self.sort_key(i, mask_type), reverse=True)
        else:
            return sorted(range(self.corrs.shape[0]),
                          key=lambda x: self.sort_key(x, mask_type), reverse=True)

    def get_neg_mask(self, threshold=0.05, as_neg_ones=False):
        cache_key = (threshold, as_neg_ones, "neg")
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        if self.binary:
            critical_ts = self.critical_ts(threshold)
            if as_neg_ones:
                ret = -1 * np.ones(self.ts.shape)
                ret[self.ts > -critical_ts] = 0
            else:
                ret = self.ts < -critical_ts
        elif self.categorical:
            raise NotImplementedError("This has not been tested yet")
            critical_fs = self.critical_fs(threshold)
            if as_neg_ones:
                ret = -1 * np.ones((self.fs.shape, self.k))
                non_sig_f_indices = np.where(self.fs < critical_fs)
                ret[non_sig_f_indices, :] = 0
                non_sig_t_indices = np.where(self.ts < stats.t.ppf(1 - threshold / 2, self.t_dofs))
                ret[non_sig_t_indices] = 0
            else:
                ret = np.array(self.fs > critical_fs)
                ret = np.logical_and(ret, self.ts < -stats.t.ppf(1 - threshold / 2, self.t_dofs))
        else:
            critical_rs = self.critical_rs(threshold)
            if as_neg_ones:
                ret = -1 * np.ones(self.corrs.shape)
                ret[self.corrs > -critical_rs] = 0
            else:
                ret = self.corrs < -critical_rs

        self.mask_cache[cache_key] = ret
        return ret

    def get_all_mask(self, threshold=0.05, as_digits=False):
        cache_key = (threshold, as_digits, "all")
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        if self.binary:
            critical_ts = self.critical_ts(threshold)
            if as_digits:
                ret = np.zeros(self.ts.shape)
                ret[self.ts > critical_ts] = 1
                ret[self.ts < -critical_ts] = -1
            else:
                ret = np.absolute(self.ts) > critical_ts
        elif self.categorical:
            raise NotImplementedError("This has not been tested yet")
            critical_fs = self.critical_fs(threshold)
            if as_digits:
                ret = np.zeros((self.fs.shape, self.k))
                non_sig_f_indices = np.where(self.fs < critical_fs)
                ret[non_sig_f_indices, :] = 0
                non_sig_t_indices = np.where(self.ts < stats.t.ppf(1 - threshold / 2, self.t_dofs))
                ret[non_sig_t_indices] = 0
                pos_sig_t_indices = np.where(self.ts > stats.t.ppf(1 - threshold / 2, self.t_dofs))
                ret[pos_sig_t_indices] = 1
                neg_sig_t_indices = np.where(self.ts < -stats.t.ppf(1 - threshold / 2, self.t_dofs))
                ret[neg_sig_t_indices] = -1
            else:
                ret = np.array(self.fs > critical_fs)
                ret = np.logical_and(ret, np.absolute(self.ts) > stats.t.ppf(1 - threshold / 2, self.t_dofs))
        else:
            critical_rs = self.critical_rs(threshold)
            if as_digits:
                ret = np.zeros(self.corrs.shape)
                ret[self.corrs > critical_rs] = 1
                ret[self.corrs < -critical_rs] = -1
            else:
                ret = np.absolute(self.corrs) > critical_rs

        self.mask_cache[cache_key] = ret
        return ret

    def count_cpm_coefficients(self, threshold, mask_type):
        mask = self.get_mask_by_type(threshold, mask_type)
        return np.count_nonzero(mask)

    def get_x(self, threshold, mask_type, x_=None):

        if x_ is None:
            x_ = self.x

        if type(x_) is list:
            x_ = np.ndarray(x_)

        if len(x_.shape) == 1:
            x_ = x_.reshape(-1, 1)

        pos_mask = self.get_pos_mask(threshold, as_ones=False)
        neg_mask = self.get_neg_mask(threshold, as_neg_ones=False)

        if self.categorical:

            raise NotImplementedError("This has not been tested yet")

            pos_masked = np.repeat(pos_mask, self.k, axis=1)
            neg_masked = np.repeat(neg_mask, self.k, axis=1)
            n_pos = np.count_nonzero(pos_masked, axis=1)
            n_neg = np.count_nonzero(neg_masked, axis=1)

            for i in range(self.k):
                pos_masked[i] = x_[:, pos_mask[:, i]]
                neg_masked[i] = x_[:, neg_mask[:, i]]
                pos_masked[i] = pos_masked[i].sum(axis=1).reshape(-1, 1)
                neg_masked[i] = neg_masked[i].sum(axis=1).reshape(-1, 1)

        else:
            pos_masked = x_[:, pos_mask]
            neg_masked = x_[:, neg_mask]
            n_pos = pos_masked.shape[1]
            n_neg = neg_masked.shape[1]

            pos_masked = pos_masked.sum(axis=1).reshape(-1, 1)
            neg_masked = neg_masked.sum(axis=1).reshape(-1, 1)

        if mask_type == "positive":
            ret = pos_masked
            n_params = n_pos
        elif mask_type == "negative":
            ret = neg_masked
            n_params = n_neg
        else:
            ret = pos_masked - neg_masked
            n_params = n_pos + n_neg

        return ret, n_params
