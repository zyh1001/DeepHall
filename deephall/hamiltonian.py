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

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from chex import ArrayTree
from jax.numpy import cos, sin, tan

from deephall.config import InteractionType, System
from deephall.types import AngularMomenta, LocalEnergy, LogPsiNetwork, OtherObservables


def coulomb_potential(r_ee: jnp.ndarray, Q: float) -> jnp.ndarray:
    """Returns the electron-electron Coulomb potential.
s
    Args:
        r_ee: The distance between two electrons.
            Shape (..., nelec, nelec).
        Q: Monopole strength. Unused.

    Returns:
        potential energy
    """
    result = jnp.sum(jnp.triu(1 / r_ee, k=1))
    return jnp.sum(jnp.triu(1 / r_ee, k=1))




def make_conf_potential(Q: float, a: jnp.ndarray, d: jnp.ndarray, r_grid, Vc_table, N_spin):
    """
    返回用于计算约束势能的函数。假设data[..., i, :] 存储第i个电子的(r,theta)。
    输入：
        Q: 磁单极子强度（可忽略，可留作占位）。
        a: 圆盘半径。
        d: 圆盘到电子平面的距离。
        N: 电子数，用于计算势能强度。
        r_grid: 一维数组（升序），存储预计算的r值节点。
        Vc_table: 一维数组，对应节点的V_c(r)值（负值）。
    返回值：
        函数 conf(data) -> 形如(...)的势能值。
    """
    def conf(data):
        # 提取径向坐标 r
        x = data[..., 0]  # 形状 (..., nelec)
        y = data[..., 1]
        r = jnp.sqrt(x**2 + y**2)
        # 插值计算 r<a*15 区域的势能
        V_small = jnp.interp(r, r_grid, Vc_table)  
        # r>=15a 时近似为点电荷势
        V_large = -N_spin / jnp.sqrt(d**2 + r**2)
        # 根据条件选择势能
        V_vals = jnp.where(r < 15*a, V_small, V_large)
        # 对所有电子求和（axis=-1对最后一个维度求和）

        result = jnp.sum(V_vals, axis=-1)
        return result  # 返回总势能
    return conf

def harmonic_potential(cos12: jnp.ndarray, Q: float) -> jnp.ndarray:
    """Returns the simple harmonic potential.

    The word "harmonic" describes the form of the Haldane pseudopotential on LLL:
        V(L) = L(L+1) / 2Q(Q+1) / sqrt(Q)
    and the corresponding real space form is:
        V(theta_12) = 1 + (Q+1) / Q * cos theta_12

    Args:
        cos12: The cosine of the angle between two electrons.
            Shape (..., nelec, nelec).
        Q: Monopole strength.

    Returns:
        potential energy
    """
    return jnp.sum(jnp.triu(1 + (Q + 1) / Q * cos12, k=1))


def make_ee_potential(
    interaction_type: InteractionType, Q: float
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create potential energy function with a given type and geometry."""
    if interaction_type == InteractionType.coulomb:
        potential_function = partial(coulomb_potential, Q=Q)
    if interaction_type == InteractionType.harmonic:
        potential_function = partial(harmonic_potential, Q=Q)

    def potential(data: jnp.ndarray) -> jnp.ndarray:
        x = data[..., 0]
        y = data[..., 1]

        cart_e = jnp.stack([x, y], axis=-1)

        cart_ee = cart_e[None] - cart_e[:, None]
        eye = jnp.eye(cart_ee.shape[0])
        r_ee = jnp.linalg.norm(cart_ee + eye[..., None], axis=-1) * (1.0 - eye)
        return potential_function(r_ee)
    
    return potential

def make_potential(
    interaction_type: InteractionType, Q: float,
    a: jnp.ndarray, d: jnp.ndarray, 
    r_grid, Vc_table, N_spin
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create potential energy function with a given type and geometry."""
    if interaction_type == InteractionType.coulomb:
        potential_function = partial(coulomb_potential, Q=Q)
    if interaction_type == InteractionType.harmonic:
        potential_function = partial(harmonic_potential, Q=Q)

    def potential(data: jnp.ndarray) -> jnp.ndarray:
        x = data[..., 0]
        y = data[..., 1]

        cart_e = jnp.stack([x, y], axis=-1)

        cart_ee = cart_e[None] - cart_e[:, None]
        eye = jnp.eye(cart_ee.shape[0])
        r_ee = jnp.linalg.norm(cart_ee + eye[..., None], axis=-1) * (1.0 - eye)
        
    
        # 提取径向坐标 r
        r = jnp.sqrt(x**2+y**2)  # 形状 (..., nelec)
        # 插值计算 r<a*15 区域的势能
        V_small = jnp.interp(r, r_grid, Vc_table)  
        # r>=15a 时近似为点电荷势
        V_large = -N_spin / jnp.sqrt(d**2 + r**2)
        # 根据条件选择势能
        V_vals = jnp.where(r < 15*a, V_small, V_large)
        # 对所有电子求和（axis=-1对最后一个维度求和）
        ee = potential_function(r_ee)
        conf = jnp.sum(V_vals, axis=-1)
        return ee+conf  # 返回总势能
    
    return potential


def make_local_kinetic_energy(f: LogPsiNetwork, Q: float, a: jnp.ndarray):
    r"""Creates a function to for the local kinetic energy.

    Args:
        f: Callable which evaluates the log of the magnitude of the wavefunction.
        Q: Flux
        r: Sphere radius

    Returns:
        Callable that evaluates the local kinetic energy, \frac{|\Lambda|^2 f}{2 R^2 f},
        where
            \frac{|\Lambda|^2 f}{f} = -\frac{\nabla^2 f}{f} + (Q \cot \theta)^2
                + 2i Q \frac{\cot \theta}{\sin \theta} \frac{\partial f}{\partial \phi},
        and
            -\frac{\nabla^2 f}{f} = - [\nabla^2 \log f + (\nabla \log f)^2].
    """

    def _lapl_over_f(
        params: ArrayTree, data: jnp.ndarray
    ) -> tuple[jnp.ndarray, AngularMomenta]:
        x, y = data[..., 0], data[..., 1]
        r_square = x**2 + y**2
        '''
        jax.debug.print("shape of data: {shape}", shape=data.shape)
        jax.debug.print("shape of r: {shape}", shape=r.shape)
        jax.debug.print("shape of theta: {shape}", shape=theta.shape)
        '''

        #        +----------------------------------------------------------+
        #        |           Prepare first and second detivatives           |
        #        +----------------------------------------------------------+


        grad_real = jax.grad(lambda p, x: f(p, x).real, argnums=1)(params, data)
        grad_imag = jax.grad(lambda p, x: f(p, x).imag, argnums=1)(params, data)
        # jax.debug.print("shape of grad_real: {shape}", shape=grad_real.shape)
        # Debug: check logpsi and gradient finite-ness
        '''
        try:
            logpsi_val = f(params, data)
            jax.debug.print("logpsi finite: {ok}, logpsi_re: {re}, logpsi_im: {im}",
                            ok=jnp.all(jnp.isfinite(logpsi_val)),
                            re=jnp.real(logpsi_val),
                            im=jnp.imag(logpsi_val))
        except Exception:
            jax.debug.print("logpsi evaluation failed")
        '''


        grad_x = grad_real[..., 0] + 1j * grad_imag[..., 0]
        grad_y = grad_real[..., 1] + 1j * grad_imag[..., 1]
        '''
        # More detailed diagnostics for grad components
        jax.debug.print(
            "grad_x finite:{ok}, nan_count:{n}, val:{val}",
            ok=jnp.all(jnp.isfinite(grad_x)),
            n=jnp.sum(jnp.isnan(grad_x)),
            val=grad_x
        )
        jax.debug.print(
            "grad_y finite:{ok}, nan_count:{n}, val:{val}",
            ok=jnp.all(jnp.isfinite(grad_y)),
            n=jnp.sum(jnp.isnan(grad_y)),
            val=grad_y
        )
        '''
        # $(\nabla \log \psi) \cdot (\nabla \log \psi)$ on a sphere
        square_grad_logpsi = jnp.sum(grad_x**2 + grad_y**2 )

        hess_real = jax.hessian(lambda p, x: f(p, x).real, argnums=1)(params, data)
        hess_imag = jax.hessian(lambda p, x: f(p, x).imag, argnums=1)(params, data)
        hess_logpsi = hess_real + 1j * hess_imag
        
        '''
        # Diagnostics for Hessian: count NaNs and report small stats
        jax.debug.print(
            "hess_real:{nr}, hess_imag:{ni}",
            nr=hess_real,
            ni=hess_imag,
        )
        '''
        #        +----------------------------------------------------------+
        #        |                Calculating kinetic energy                |
        #        +----------------------------------------------------------+

        # $\nabla^2 \log \psi$ on disk
        grad_grad_logpsi = jnp.sum(
            jnp.diagonal(hess_logpsi[:, 0, :, 0])
            + jnp.diagonal(hess_logpsi[:, 1, :, 1])
        )
        # jax.debug.print("grad_grad_logpsi finite:{ok}, value:{v}", ok=jnp.isfinite(grad_grad_logpsi), v=grad_grad_logpsi)
        # See section 3.10.3 of "Composite Fermions"
        magnetic_contribution = jnp.sum(
            1j*y*grad_x - 1j*x*grad_y + r_square / 4
        )
        # jax.debug.print("magnetic_contribution finite:{ok}, val:{v}", ok=jnp.all(jnp.isfinite(magnetic_contribution)), v=jnp.nan_to_num(magnetic_contribution))
        sum_kinetic_momentum_square = (
            -grad_grad_logpsi - square_grad_logpsi + magnetic_contribution
        )
        '''
        jax.debug.print(
            "square_grad_logpsi finite:{ok}, val:{v}", ok=jnp.all(jnp.isfinite(square_grad_logpsi)), v=jnp.nan_to_num(square_grad_logpsi)
        )
        '''
        kinetic_energy = sum_kinetic_momentum_square / 2
        # Debug: print kinetic stats
        '''
        jax.debug.print("kinetic finite: {ok}, val: {val}",
                ok=jnp.all(jnp.isfinite(kinetic_energy)),
                val=jnp.real(kinetic_energy)
        )
        '''
        #        +----------------------------------------------------------+
        #        |        Calculating angular momentum square (L^2)         |
        #        +----------------------------------------------------------+


        # Note that theta_hat_prime alrealdy has a 1/sin factor
        # magnetic_term = r_square**2 * Q**2/ a**4 + 2j*r_square*Q*(-x*grad_y + y*grad_x) / a**2
        grad_grad_x_psi  = grad_x**2 + jnp.diagonal(hess_logpsi[:, 0, :, 0])
        grad_grad_y_psi = grad_y**2 + jnp.diagonal(hess_logpsi[:, 1, :, 1])
        grad_grad_x_y_psi = jnp.diagonal(hess_logpsi[:, 0, :, 1]) + grad_x * grad_y
        # We first assume everything commutes, and add back extra terms at the end
        angular_momentum_square = jnp.sum(
             - x**2 * grad_grad_y_psi - y**2 * grad_grad_x_psi
              + x * grad_x + y * grad_y + 2*x*y*grad_grad_x_y_psi 
        )
# 总 Lz^2 的局部值
# 利用关系: (L^2 psi)/psi = div(L psi)/psi ... 或者直接展开
# 对于 Lz 是对易的，最简单的方法是计算局部算符的平方项和导数项
# 但更稳健的方法是利用总算符：
# Lz_total_sq_local = (Lz_total_local)^2 + sum_i (-i(x_i d/dy_i - y_i d/dx_i) lz_local_each_i)
# 考虑到代码实现的复杂性，如果波函数是 Lz 的本征态，
# 那么 (Lz_total_local)^2 应该已经等于 Lz_total_square 的局部值。
        #        +----------------------------------------------------------+
        #        |                     Assemble outputs                     |
        #        +----------------------------------------------------------+
        # total Lz = -i D, where D = sum (x_i d/dy_i - y_i d/dx_i)
        D_f = jnp.sum(x * grad_y - y * grad_x)
        term1 = jnp.sum(jnp.outer(x, x) * hess_logpsi[:, 1, :, 1])
        term2 = - jnp.sum(jnp.outer(x, y) * hess_logpsi[:, 1, :, 0])
        term3 = - jnp.sum(jnp.outer(y, x) * hess_logpsi[:, 0, :, 1])
        term4 = jnp.sum(jnp.outer(y, y) * hess_logpsi[:, 0, :, 0])
        term5 = - jnp.sum(x * grad_x + y * grad_y)
        D2_f = term1 + term2 + term3 + term4 + term5
        angular_momentum_z_square = - (D_f**2 + D2_f)
        
        angular_momentum_z=jnp.sum(x * grad_y.imag - y * grad_x.imag)
        other_observables = AngularMomenta(
            angular_momentum_z=angular_momentum_z.real,
            angular_momentum_z_square=angular_momentum_z_square.real,
        )
        return kinetic_energy, other_observables

    return _lapl_over_f

# FIXME
def local_energy(f: LogPsiNetwork, system: System) -> LocalEnergy:
    """Creates the function to evaluate the local energy.

    Args:
        f: Callable which returns the sign and log of the magnitude of the
            wavefunction given the network parameters and configurations data.
        system: Config for system.

    Returns:
        Callable with signature e_l(params, key, data) which evaluates the local
        energy of the wavefunction given the parameters params, RNG state key,
        and a single MCMC configuration in data.
    """
    Q = system.flux
    d = system.d
    N = system.nspins[0] + system.nspins[1]
    radius = jnp.array(system.radius or jnp.sqrt(2*Q))
    ke = make_local_kinetic_energy(f, Q, radius)
    pe = make_ee_potential(system.interaction_type, Q)

    Vc = jnp.load(f"Vc_n{N}_q{Q}.npy")
    r_grid = jnp.load(f"r_n{N}_q{Q}.npy")
    pc = make_conf_potential(Q, radius, d, r_grid, Vc, N)
    def _e_l(
        params: ArrayTree, data: jnp.ndarray
    ) -> tuple[jnp.ndarray, OtherObservables]:
        """Returns the total energy.

        Args:
            params: network parameters.
            data: MCMC configuration.

        Returns:
            Local energy and other observables.
        """
        # FIXME
        ee_potential = pe(data) * system.interaction_strength
        # jax.debug.print("ee_potential:{}", ee_potential)
        #FIXME
        conf_potential = pc(data) * system.interaction_strength
        # jax.debug.print("conf_potential:{}", conf_potential )
        potential = ee_potential + conf_potential
        kinetic, angular_momenta = ke(params, data)
        # jax.debug.print("kinetic:{}", kinetic)
        return kinetic + potential, angular_momenta | {
            "potential": potential,
            "kinetic": kinetic,
        }
    

    return _e_l
