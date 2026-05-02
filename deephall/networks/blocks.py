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

import numpy as np
from flax import linen as nn
from jax import numpy as jnp
import jax
from scipy import special as ss

from deephall.config import OrbitalType, EnvelopeType

# NO CHANGE
class FeaturedOrbitals(nn.Module):
    nspins: tuple[int, int]
    features: list[int]

    @nn.compact
    def __call__(self, h_one):
        orbital_list = [
            nn.DenseGeneral(self.features)(h_one_alpha)
            + 1j * nn.DenseGeneral(self.features)(h_one_alpha)
            for h_one_alpha in jnp.split(h_one, (self.nspins[0],))
            if len(h_one_alpha)
        ]
        return jnp.concat(orbital_list) # [nspins, Q+1,nspins,1]


class Orbitals(nn.Module):
    type: OrbitalType
    Q: float
    nspins: tuple[int, int]
    ndets: int
    envelope_type: EnvelopeType
    def setup(self):
        
        if self.type == OrbitalType.full:
            self.featured_orbitals = FeaturedOrbitals(
                nspins=self.nspins,
                features=(int(self.Q + 1), sum(self.nspins), self.ndets),
            )
        elif self.type == OrbitalType.sparse:
            self.featured_orbitals = FeaturedOrbitals(
                nspins=self.nspins,
                features=(8, sum(self.nspins), self.ndets),
            )
            self.lll_weight = nn.DenseGeneral(int(self.Q + 1), axis=1)
            
    @nn.compact
    def __call__(self, h_one, x, y):
        orbitals = self.featured_orbitals(h_one) # [nspins,Q+1,nspins]
        # Diagnostic: check normalization factors
        '''
        jax.debug.print(
            "norm_factor finite: {ok}, min: {mn}, max: {mx}",
            ok=jnp.all(jnp.isfinite(self.norm_factor)),
            mn=jnp.min(jnp.nan_to_num(self.norm_factor)),
            mx=jnp.max(jnp.nan_to_num(self.norm_factor)),
        )
        '''
        if self.type == OrbitalType.sparse:
            orbitals = self.lll_weight(orbitals).transpose((0, 3, 1, 2))


        r_square= x**2+y**2
        r_square = r_square[..., None]
        
        # --- 修改部分开始 ---
        # 1. 轨道总数为 Q + 1
        num_orbitals = int(self.Q + 1)
        # 2. 将 sigma 的 shape 指定为 (num_orbitals,) 而不是 (1,)
        sigma = self.param("sigma", nn.initializers.constant(0.25), (num_orbitals,))
        
        if self.envelope_type == EnvelopeType.no_m_learnable:
            # r_square shape 是 (batch, 1), sigma 是 (Q+1,)
            # 相乘后自动发生广播（Broadcasting），envelope 的 shape 变为 (batch, Q+1)
            envelope = jnp.exp(- r_square * jnp.abs(sigma))
            
        elif self.envelope_type == EnvelopeType.no_m_fixed:
            envelope = jnp.exp(- r_square * 0.25)
        
        elif self.envelope_type == EnvelopeType.m_learnable:
            m = np.arange(0, num_orbitals)
            norm_factor = np.array(1.0 / np.sqrt(ss.factorial(m)))
            z = x - 1j * y
            z = z[..., None]
            # 这里 z**m 同样会产生形状为 (batch, Q+1) 的数组
            envelope = norm_factor * z ** m * jnp.exp(- r_square * sigma)
            
        elif self.envelope_type == EnvelopeType.m_fixed:
            m = np.arange(0, num_orbitals)
            norm_factor = np.array(1.0 / np.sqrt(ss.factorial(m)))
            z = x - 1j * y
            z = z[..., None]
            envelope = norm_factor * z ** m * jnp.exp(- r_square * 0.25)
        # --- 修改部分结束 ---

        # 此时 envelope shape 为 (batch, Q+1)，
        # envelope[..., None, None] 会在其末尾新增两个维度变为 (batch, Q+1, 1, 1)，
        # 能完美地与 orbitals 进行乘法广播然后再基于 axis=1 求和。
        orbitals = jnp.sum(orbitals * envelope[..., None, None], axis=1)
        
        
        '''
        jax.debug.print("print shapes in \'Orbitals\'")
        jax.debug.print("shape of original obitals:{shape}", shape = orbitals.shape)
        '''
        '''
        jax.debug.print("shape of h_one:{shape}", shape = h_one.shape)
        jax.debug.print("shape of r:{shape}", shape=r.shape)
        jax.debug.print("shape of m:{shape}", shape=m.shape)
        jax.debug.print("shape of z:{shape}", shape=z.shape)
        jax.debug.print("shape of r_norm:{shape}", shape=r_norm.shape)
        jax.debug.print("shape of envelope:{shape}", shape = envelope.shape)
        jax.debug.print("shape of modified obitals:{shape}", shape = orbitals.shape)
        '''
        return jnp.moveaxis(orbitals, -1, 0)  # Move ndets dim to the front


class Jastrow(nn.Module):
    nspins: tuple[int, int]

    @nn.compact
    def __call__(self, electrons: jnp.ndarray) -> jnp.ndarray:
        nspins = self.nspins
        r_ee = self.calculated_r_ee(electrons)
        r_ees = [
            jnp.split(r, nspins[0:1], axis=1)
            for r in jnp.split(r_ee, nspins[0:1], axis=0)
        ]
        r_ees_parallel = jnp.concatenate(
            [
                r_ees[0][0][jnp.triu_indices(nspins[0], k=1)],
                r_ees[1][1][jnp.triu_indices(nspins[1], k=1)],
            ]
        )

        if r_ees_parallel.shape[0] > 0:
            alpha_par = self.param("ee_par", nn.initializers.ones, (1,))
            beta_par = self.param("beta_par", nn.initializers.constant(0.25), (1,))
            a_p = jnp.squeeze(alpha_par)
            b_p = jnp.squeeze(beta_par)
            jastrow_ee_par = jnp.sum(
                -(b_p * a_p**2) / (a_p + r_ees_parallel)
            )
        else:
            jastrow_ee_par = jnp.asarray(0.0)

        if r_ees[0][1].shape[0] > 0:
            alpha_anti = self.param("ee_anti", nn.initializers.ones, (1,))
            beta_anti = self.param("beta_anti", nn.initializers.constant(0.5), (1,))
            a_a = jnp.squeeze(alpha_anti)
            b_a = jnp.squeeze(beta_anti)
            jastrow_ee_anti = jnp.sum(
                -(b_a * a_a**2) / (a_a + r_ees[0][1])
            )
        else:
            jastrow_ee_anti = jnp.asarray(0.0)

        total_jastrow = jastrow_ee_anti + jastrow_ee_par
        '''
        jax.debug.print(
            "jastrow finite: {ok}, jastrow_total: {v}",
            ok=jnp.all(jnp.isfinite(total_jastrow)),
            v=jnp.nan_to_num(total_jastrow),
        )
        '''

        return total_jastrow

    def calculated_r_ee(self, electrons: jnp.ndarray) -> jnp.ndarray:

        x = electrons[..., 0]
        y = electrons[..., 1]

        cart_e = jnp.stack([x, y], axis=-1)

        cart_ee = cart_e[None] - cart_e[:, None]
        eye = jnp.eye(cart_ee.shape[0])
        return jnp.linalg.norm(cart_ee + eye[..., None], axis=-1) * (1.0 - eye)