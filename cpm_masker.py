import numpy as np
from scipy import stats


class CPMMasker:
    def __init__(self, x_, y_, corrs_=None, ts_=None, binary=False):

        self.mask_cache = {}
        self.corrs = corrs_
        self.ts = ts_
        self.dofs = None

        if binary:
            if self.ts is None:
                categories = np.unique(y_)
                if len(categories) != 2:
                    raise ValueError("Binary classification requires two classes")
                y_ = np.array([categories.tolist().index(y) for y in y_])
                zero_means = np.mean(x_[y_ == 0], axis=0)
                one_means = np.mean(x_[y_ == 1], axis=0)
                zero_sds = np.std(x_[y_ == 0], axis=0)
                one_sds = np.std(x_[y_ == 1], axis=0)

                self.ts = (one_means - zero_means) / np.sqrt(
                    one_sds ** 2 / y_.sum() + zero_sds ** 2 / (y_.shape[0] - y_.sum()))
                self.dofs = (one_sds ** 2 / y_.sum() + zero_sds ** 2 / (y_.shape[0] - y_.sum())) ** 2 / (
                        (one_sds ** 2 / y_.sum()) ** 2 / (y_.sum() - 1) + (
                        zero_sds ** 2 / (y_.shape[0] - y_.sum())) ** 2 / (
                                (y_.shape[0] - y_.sum()) - 1))
        else:
            if self.corrs is None:
                y_corr = (y_ - y_.mean()) / np.linalg.norm(y_)
                x_corr = (x_ - x_.mean(axis=0).reshape(1, -1)) / np.linalg.norm(x_, axis=0).reshape(1, -1)
                x_corr = np.nan_to_num(x_corr)
                self.corrs = x_corr.T @ y_corr
                self.dofs = np.ones(y_.shape[0]) * (x_.shape[0] - 2)

        self.binary = binary
        self.x = x_
        self.y = y_

    def clone(self):
        return CPMMasker(self.x, self.y, self.corrs)

    def critical_ts(self, alpha):
        return np.array([stats.t.ppf(1 - alpha / 2, dof) for dof in self.dofs])

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
