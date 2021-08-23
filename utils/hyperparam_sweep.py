# !/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# import torch.multiprocessing as multiprocessing
# # We must import this explicitly, it is not imported by the top-level
# # multiprocessing module.
# from torch.multiprocessing import Pool as OldPool
# import time
#
# from random import randint
#
#
# class NoDaemonProcess(multiprocessing.Process):
#     # make 'daemon' attribute always return False
#     def _get_daemon(self):
#         return False
#
#     def _set_daemon(self, value):
#         pass
#
#     daemon = property(_get_daemon, _set_daemon)
#
#
# # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# # because the latter is only a wrapper function, not a proper class.
# class MyPool(OldPool):
#     Process = NoDaemonProcess
#
#
# def sleepawhile(t):
#     print("Sleeping %i seconds..." % t)
#     time.sleep(t)
#     return t
#
#
# def work(num_procs):
#     print("Creating %i (daemon) workers and jobs in child." % num_procs)
#     pool = multiprocessing.Pool(num_procs)
#
#     result = pool.map(sleepawhile,
#                       [randint(1, 5) for x in range(num_procs)])
#
#     # The following is not really needed, since the (daemon) workers of the
#     # child's pool are killed when the child is terminated, but it's good
#     # practice to cleanup after ourselves anyway.
#     pool.close()
#     pool.join()
#     return result
#
#
# def test():
#     print("Creating 5 (non-daemon) workers and jobs in main process.")
#     pool = MyPool(5)
#
#     result = pool.map(work, [randint(1, 5) for x in range(5)])
#
#     pool.close()
#     pool.join()
#     print(result)


def create_logarithmic_lattice(min_val, max_val, num_points):
    ratio = (max_val / min_val) ** (1 / (num_points - 1))
    points = [min_val * (ratio ** i) for i in range(num_points)]
    assert len(points) == num_points
    return points
