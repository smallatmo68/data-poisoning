import numpy as np
import pandas as pd
import sys


class InfluenceErrorSignals:
    def __init__(self):
        self.__self_influence = {
            "SI": self.si_fast,
        }
        self.__marginal_signals = {
            "MPI": self.mpi_fast,
            "MAI": self.mai_fast,
            "MI": self.mi_fast,
            "AAI": self.aai_fast
        }
        self.__conditional_signals = {
            "NIL": self.nil_fast,
            "NIL_opt": self.nil_opt_fast,
            "#SLIP": self.num_slip_fast,
            "#SLIN": self.num_slin_fast,
            "GD-class": self.gd_class_fast
        }

    def train_signals_names(self):
        return {
            *self.__self_influence.keys(),
            *self.__marginal_signals.keys(),
            *self.__conditional_signals.keys(),
        }

    def compute_train_signals_fast(
        self,
        self_inf_arr,
        train_to_test_inf_mat,
        y_train,
        y_test,
        silent=False
    ):
        signal_vals = {}
        for sig_name in self.train_signals_names():
            if not silent:
                print(sig_name)
            if sig_name in self.__self_influence:
                signal_vals[sig_name] = self_inf_arr
            elif sig_name in self.__marginal_signals:
                ms, _ = self.__marginal_signals[sig_name](train_to_test_inf_mat)
                signal_vals[sig_name] = ms
            elif sig_name in self.__conditional_signals:
                cs, _ = self.__conditional_signals[sig_name](
                    train_to_test_inf_mat, y_train, y_test
                )
                signal_vals[sig_name] = cs
        return pd.DataFrame.from_dict(signal_vals)

    def si_fast(self, self_inf_mat):
        return np.diagonal(self_inf_mat)

    def nil_fast(
        self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        nil_values = np.zeros(len(y_train))
        nil_values_var = np.zeros(len(y_train))
        for l in np.unique(y_train):
            l_train_ids = np.where(y_train == l)[0]
            l_test_ids = np.where(y_test == l)[0]
            inf_of_l_samples = train_test_inf_mat_tmp[l_train_ids, :]
            inf_of_l_samples_on_test_l_samples = inf_of_l_samples[:, l_test_ids]
            inf_of_l_samples_on_test_l_samples[
                inf_of_l_samples_on_test_l_samples > 0
            ] = 0
            nil_values[l_train_ids] = inf_of_l_samples_on_test_l_samples.sum(axis=1)
            nil_values_var[l_train_ids] = inf_of_l_samples_on_test_l_samples.var(axis=1)
        return -1 * nil_values, nil_values_var

    def gd_class_fast(
        self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        htil_values = None
        for l in np.unique(y_train):
            l_test_ids = np.where(y_test == l)[0]
            inf_of_l_samples_on_test_l_samples = train_test_inf_mat_tmp[:, l_test_ids]
            htil_values_tmp = inf_of_l_samples_on_test_l_samples.sum(axis=1)
            if htil_values is None:
                htil_values = htil_values_tmp.copy()
            else:
                htil_values = np.vstack((htil_values, htil_values_tmp))
        return -htil_values.min(axis=0), htil_values.argmin(axis=0)

    def nil_opt_fast(
        self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        nil_values = None
        for l in np.unique(y_train):
            l_test_ids = np.where(y_test == l)[0]
            inf_of_l_samples_on_test_l_samples = train_test_inf_mat_tmp[:, l_test_ids]
            inf_of_l_samples_on_test_l_samples[
                inf_of_l_samples_on_test_l_samples > 0
            ] = 0
            nil_values_tmp = inf_of_l_samples_on_test_l_samples.sum(axis=1)
            if nil_values is None:
                nil_values = nil_values_tmp.copy()
            else:
                nil_values = np.vstack((nil_values, nil_values_tmp))
        return -1 * nil_values.min(axis=0), nil_values.argmin(axis=0)

    def mpi_fast(self, train_test_inf_mat: np.ndarray):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        train_test_inf_mat_tmp[train_test_inf_mat_tmp < 0] = 0
        return train_test_inf_mat_tmp.sum(axis=1), train_test_inf_mat_tmp.var(axis=1)

    def mai_fast(self, train_test_inf_mat: np.ndarray):
        return np.absolute(train_test_inf_mat).sum(axis=1), train_test_inf_mat.var(
            axis=1
        )

    def aai_fast(self, train_test_inf_mat: np.ndarray):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        return np.absolute(train_test_inf_mat_tmp).mean(axis=1), train_test_inf_mat_tmp.var(
            axis=1
        )

    def mi_fast(self, train_test_inf_mat: np.ndarray):
        return train_test_inf_mat.sum(axis=1), train_test_inf_mat.var(
            axis=1
        )

    def num_slip_fast(
        self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        slip_values = np.zeros(len(y_train))
        slip_values_var = np.zeros(len(y_train))
        for l in np.unique(y_train):
            l_train_ids = np.where(y_train == l)[0]
            l_test_ids = np.where(y_test == l)[0]
            inf_of_l_samples = train_test_inf_mat_tmp[l_train_ids, :]
            inf_of_l_samples_on_test_l_samples = inf_of_l_samples[:, l_test_ids]
            slip_values[l_train_ids] = np.sum(
                np.array(inf_of_l_samples_on_test_l_samples) > 0, axis=1
            )
            slip_values_var[l_train_ids] = np.var(
                np.array(inf_of_l_samples_on_test_l_samples) > 0, axis=1
            )
        return slip_values, slip_values_var

    def num_slin_fast(
        self, train_test_inf_mat: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
    ):
        train_test_inf_mat_tmp = train_test_inf_mat.copy()
        slin_values = np.zeros(len(y_train))
        slin_values_var = np.zeros(len(y_train))
        for l in np.unique(y_train):
            l_train_ids = np.where(y_train == l)[0]
            l_test_ids = np.where(y_test == l)[0]
            inf_of_l_samples = train_test_inf_mat_tmp[l_train_ids, :]
            inf_of_l_samples_on_test_l_samples = inf_of_l_samples[:, l_test_ids]
            slin_values[l_train_ids] = np.sum(
                np.array(inf_of_l_samples_on_test_l_samples) < 0, axis=1
            )
            slin_values_var[l_train_ids] = np.var(
                np.array(inf_of_l_samples_on_test_l_samples) < 0, axis=1
            )
        return slin_values, slin_values_var
