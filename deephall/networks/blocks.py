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
from scipy import special as ss

from deephall.config import OrbitalType

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
        return jnp.concat(orbital_list)


class Orbitals(nn.Module):
    type: OrbitalType
    Q: int
    nspins: tuple[int, int]
    ndets: int

    def setup(self):
        m = np.arange(0, int(self.Q + 1))
        self.norm_factor = jnp.array(1.0 / np.sqrt(ss.factorial(m)))
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

    def __call__(self, h_one, r, theta):
        orbitals = self.featured_orbitals(h_one)
        if self.type == OrbitalType.sparse:
            orbitals = self.lll_weight(orbitals).transpose((0, 3, 1, 2))

        m = jnp.arange(0, int(self.Q + 1))
        z = r * jnp.exp(1j  * theta)
        z = z[..., None]
        envelope = self.norm_factor * z ** m * jnp.exp(- jnp.abs(z)**2 / 4)
        orbitals = jnp.sum(orbitals * envelope[..., None, None], axis=1)

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
            jastrow_ee_par = jnp.sum(
                -(0.25 * alpha_par**2) / (alpha_par + r_ees_parallel)
            )
        else:
            jastrow_ee_par = jnp.asarray(0.0)

        if r_ees[0][1].shape[0] > 0:
            alpha_anti = self.param("ee_anti", nn.initializers.ones, (1,))
            jastrow_ee_anti = jnp.sum(
                -(0.5 * alpha_anti**2) / (alpha_anti + r_ees[0][1])
            )
        else:
            jastrow_ee_anti = jnp.asarray(0.0)

        return jastrow_ee_anti + jastrow_ee_par

    def calculated_r_ee(self, electrons: jnp.ndarray) -> jnp.ndarray:

        r = electrons[..., 0]
        theta = electrons[..., 1]

        # polar -> cartesian
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)

        cart_e = jnp.stack([x, y], axis=-1)

        # pairwise vectors, shape (..., nelec, nelec, 2)
        cart_ee = cart_e[..., :, None, :] - cart_e[..., None, :, :]
        
        distances = jnp.linalg.norm(cart_ee, axis=-1)
        distances = jnp.maximum(distances, 1e-8)  # 防止除以 0
        return distances        