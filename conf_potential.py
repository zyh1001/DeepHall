# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import jax
import jax.scipy.special as jsp
from scipy.special import ellipk
from scipy.integrate import quad
import os
import logging
import numpy as np
from numpy.polynomial.legendre import leggauss


'''
Pre-compute the confining potential
'''
def make_gauss_nodes(a:float, n_node: int=10000):

    x, w = leggauss(n_node) # x in [-1,1]

    # map to [0,a]
    rp = 0.5 * a * (x + 1.0)
    wp = 0.5 * a * w

    return rp.astype(np.float32), wp.astype(np.float32)


def gausse(r_vals, rp, wp, a, d: float, N_spins: int):

    """
    r_vals: (Nr,)
    rp: (Ng,)
    wp: (Ng,)
    """

    r = r_vals[:, None]      # (Nr,1)
    rp = rp[None, :]         # (1,Ng)

    A = (r + rp)**2 + d**2
    B = (r - rp)**2 + d**2
    m1 = 4 * r * rp / A
    m2 = 4 * r * rp / B
    k1 = ellipk(m1)
    k2 = ellipk(-m2)
    # m1 = np.clip(m1, None, 1.0)   # 数值稳定
    # m2 = np.clip(m2, None, 1.0)
    f = 2*rp * (k1 / np.sqrt(A) 
              +  k2 / np.sqrt(B))

    integral = np.sum(wp * f, axis=1)

    pref = - N_spins / (np.pi * a**2)

    return pref * integral


def single_quad(r: float, N_spins: int, a: float, d: float=0.1, 
                limit: int=1000, epsrel: float=1e-4):


    def integrand(rp, r, d):
        """
        被积函数 f(r,r')
        """

        A = (r + rp)**2 + d**2
        B = (r - rp)**2 + d**2
        m1 = 4 * r * rp / A
        m2 = 4 * r * rp / B
        k1 = ellipk(m1)
        k2 = ellipk(-m2)
        # m1 = np.clip(m1, None, 1.0)   # 数值稳定
        # m2 = np.clip(m2, None, 1.0)
        f = 2*rp * (k1 / np.sqrt(A) 
                +  k2 / np.sqrt(B))

        return f
    
    f = lambda rp: integrand(rp, r, d)
    I, err = quad(f, 0.0, a, epsrel=epsrel, limit=limit)
    pref = - N_spins / np.pi / a**2

    return pref * I, err

def quad_array(N_spins, r_vals, a: float, d: float=0.1, 
         limit: int=1000, epsrel: float=1e-4):
    Vc = np.zeros_like(r_vals)
    err = np.zeros_like(r_vals)
    i=0
    for r in r_vals:

        val, e = single_quad(r, N_spins, a, d, limit, epsrel)

        Vc[i] = val
        err[i] = e

        if i % 10 == 0:

            print(f"{i}/{len(r_vals)}  err={e:.2e}")
        i+=1
    return Vc, err


def pre_compute(N_spins: int, Q: int, d: float=0.1, n_node: int=10000, 
                limit: int=1000, epsrel: float=1e-4, use_quad = False):
    a = np.sqrt(2 * Q)
    r_vals = np.linspace(0.0, 15*a, n_node)
    
    current_file_directory = os.path.dirname(__file__)
    path_vc = os.path.join(current_file_directory,f"Vc_n{N_spins}_q{Q}.npy")
    path_r = os.path.join(current_file_directory,f"r_n{N_spins}_q{Q}.npy")

    if use_quad == False:
        rp, wp = make_gauss_nodes(a, n_node)
        Vc = gausse(r_vals, rp, wp, a, d, N_spins)

    else:
        Vc , err = quad_array(N_spins, r_vals, a, d, limit, epsrel)
        err_np = np.array(err)
        path_err = os.path.join(current_file_directory,f"err_n{N_spins}_q{Q}.npy")
        np.save(path_err, err_np)
    # 转 numpy 保存
    Vc_np = np.array(Vc)
    # 获取当前文件所在的目录

    np.save(path_vc, Vc_np)
    np.save(path_r, np.array(r_vals))

    if use_quad == True:
        err = np.array

    if os.path.exists(path_vc):
        logging.info("Data saved")
    else:
        logging.error("Can\'t save data")
if __name__ == "__main__":
    logging.info("Pre-computing confing potential")
    pre_compute(6, 18, 0.1, use_quad = True)