from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from .rec_init import matrix_init, truncated_normal_matrix_init, theta_init, nu_init, gamma_log_init, symmetric_matrix_init
from flax.core.frozen_dict import unfreeze


# Parallel scan operations
@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


@jax.vmap
def binary_operator_diag_spatial(q_i, q_j):
    """Same as above but stop the gradient for the recurrent connection"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, jax.lax.stop_gradient(A_j * b_i) + b_j


class LRU(nn.Module):
    """
    LRU layer that updates internal elegibility traces to allow online learning.
    """

    d_hidden: int  # hidden state dimension
    d_model: int  # input and output dimensions
    seq_length: int  # time sequence length
    gamma_norm: bool = True  # use gamma normalization
    exp_param: bool = True  # exponential parametrization for lambda
    r_min: float = 0.0  # smallest eigenvalue norm
    r_max: float = 1.0  # largest eigenvalue norm
    max_phase: float = 6.28  # max phase eigenvalue
    training_mode: str = "bptt"  # which learning algorithm that will be used
    training: bool = False  # TODO remove, for debugging purposes
    has_layer_output: bool = True
    mixing: str = "full"  # type of mixing for the two hidden states
    d_in: int = None  # input dimension, if not none, it means we are instantiating the first layer
    d_out: int = None  # output dimension, if not none, it means we are instantiating the last layer
    has_non_linearity_in_recurrence: bool = False  # whether to apply a non-linearity after the recurrent step

    def _rotate_pair_to_eigen(self, x):
        """
        x: array with shape (2, self.d_hidden, *extra_dims)
        - axis 0: src/dst
        - axis 1: hidden feature index j
        - remaining axes: anything (lambda index, gamma index, d_model, ...)
        Applies R(φ_j)^T to the (src,dst) pair for each feature j,
        broadcasting over extra_dims.
        """
        original_shape = x.shape
        # Collapse all extra dims into one
        x = x.reshape(2, self.d_hidden, -1)        # (2, H, F)
        x_s, x_d = x[0], x[1]                      # each (H, F)

        phi = self.phi                             # (H,)
        c = jnp.cos(phi)[:, None]                  # (H, 1), broadcasts over F
        s = jnp.sin(phi)[:, None]

        # z = R^T h
        z1 = c * x_s - s * x_d                    # (H, F)
        z2 = s * x_s + c * x_d                    # (H, F)

        z = jnp.stack([z1, z2], axis=0)           # (2, H, F)
        return z.reshape(original_shape)          # back to (2, H, *extra_dims)


    def _rotate_pair_from_eigen(self, z):
        """
        Inverse of _rotate_pair_to_eigen:
        z: shape (2, H, *extra_dims) in eigen basis,
        returns (2, H, *extra_dims) in node basis.
        """
        original_shape = z.shape
        z = z.reshape(2, self.d_hidden, -1)        # (2, H, F)
        z1, z2 = z[0], z[1]                        # (H, F)

        phi = self.phi
        c = jnp.cos(phi)[:, None]
        s = jnp.sin(phi)[:, None]

        # h = R z
        h_s = c * z1 + s * z2                     # (H, F)
        h_d = -s * z1 + c * z2                    # (H, F)

        h = jnp.stack([h_s, h_d], axis=0)         # (2, H, F)
        return h.reshape(original_shape)


    def _to_eigen_basis(self, h_flat):
        h = h_flat.reshape(2, self.d_hidden)            # (2, H)
        z = self._rotate_pair_to_eigen(h[:, :, None])   # (2, H, 1)
        return z[:, :, 0].reshape(-1)                   # back to (2*H,)


    def _from_eigen_basis(self, z_flat):
        z = z_flat.reshape(2, self.d_hidden)              # (2, H)
        h = self._rotate_pair_from_eigen(z[:, :, None])   # (2, H, 1)
        return h[:, :, 0].reshape(-1)                     # back to (2*H,)



    def get_diag_lambda(self, nu=None, theta=None):
        """
        Transform parameters nu and theta into the diagonal of the recurrent
        Lambda matrix.

        Args:
            nu, theta array[N]: when set to their default values, None, the
                parameters will take the values of the Module.
                NOTE: these arguments are added in order to backpropagate through this
                transformation.
        """
        if nu is None:
            nu = self.nu
        if theta is None:
            theta = self.theta
        if self.exp_param:
            theta = jnp.exp(theta)
            nu = jnp.exp(nu)
        if self.mixing in ["none", "rotational"]:
            nu = nu.reshape(4, -1)
            theta = theta.reshape(4, -1)
            return (jnp.exp(-nu + 1j * theta) * jnp.array([1, 0, 0, 1]).reshape(4, 1)).reshape(-1)
        return (jnp.exp(-nu + 1j * theta)).reshape(-1)

    def get_diag_gamma(self):
        """
        Transform parameters gamma_log into the diagonal terms of the modulation matrix gamma.
        """
        if self.gamma_norm:
            return jnp.exp(self.gamma_log)
        else:
            return jnp.ones((self.d_hidden,))

    def get_B(self):
        """
        Get input to hidden matrix B.
        """
        return self.B_re + 1j * self.B_im

    def get_B_norm(self):
        """
        Get modulated input to hidden matrix gamma B.
        """
        return self.get_B() * jnp.expand_dims(self.get_diag_gamma(), axis=-1)
    
    def to_output_single_step(self, inputs, hidden_states):
        """
        Compute output given inputs and hidden states.

        Args:
            inputs array[T, H].
            hidden_states array[T, N].
        """

        C = self.C_re + 1j * self.C_im
        D = self.D

        y = (C @ hidden_states).real + D * inputs
        return y
    
    def get_single_step_hidden_states(self, inputs, old_hidden_states):
        """
        Compute the hidden states corresponding to inputs

        Return:
            hidden_states array[T, N]
        """
        # Materializing the diagonal of Lambda and projections
        diag_lambda = self.get_diag_lambda()
        B_norm = self.get_B_norm()

        # Running the LRU
        Bu = B_norm @ inputs

        if self.training_mode == "bptt":

            if self.mixing in ["full", "symmetric", "none", "rotational"]:
                diag_lambda = diag_lambda.reshape(2, 2, self.d_hidden)
                old_hidden_states = old_hidden_states.reshape(1, 2, -1)
                lambda_states = jnp.sum(diag_lambda * old_hidden_states, axis=1).reshape(-1)
            else:
                raise ValueError("Mixing type not recognized")
        else:
            
            if self.mixing in ["full", "symmetric", "none", "rotational"]:
                diag_lambda = diag_lambda.reshape(2, 2, self.d_hidden)
                old_hidden_states = old_hidden_states.reshape(1, 2, -1)
                lambda_states = jnp.sum(diag_lambda * jax.lax.stop_gradient(old_hidden_states), axis=1).reshape(-1)
            else:
                raise ValueError("Mixing type not recognized")

        hidden_states = self.normalizer * lambda_states + Bu

        return hidden_states

    def setup(self):
        # Check that desired approximation is handled
        if self.training_mode == "online_snap1":
            raise NotImplementedError("SnAp-1 not implemented for LRU")
        assert self.training_mode in [
            "bptt",
            "online_full",
            "online_full_rec",
            "online_full_rec_simpleB",
            "online_snap1",  # same as online_full
            "online_spatial",
            "online_1truncated",
            "online_reservoir",
        ]
        self.online = "online" in self.training_mode  # whether we compute the gradient online
        if self.online:
            self.approximation_type = self.training_mode[7:]

        # NOTE if exp_param is true, self.theta and self.nu actually represent the log of nu and
        # theta lambda is initialized uniformly in complex plane
            
        self.theta = self.param(
            "theta",
            partial(theta_init, max_phase=self.max_phase, log=self.exp_param),
            (4 * self.d_hidden,),
        )  # phase of lambda in [0, max_phase]
        self.nu = self.param(
            "nu",
            partial(nu_init, r_min=self.r_min, r_max=self.r_max, log=self.exp_param),
            (4 * self.d_hidden,),
        )  # norm of lambda in [r_min, r_max]
        if self.gamma_norm:
            self.gamma_log = self.param(
                "gamma_log", partial(gamma_log_init, log=self.exp_param, mixing=self.mixing), (self.nu, self.theta)
            )

        # Glorot initialized Input/Output projection matrices
        if self.d_in is not None:
            self.B_re = self.param(
                "B_re",
                partial(matrix_init, normalization=jnp.sqrt(self.d_in)),
                (2*self.d_hidden, self.d_in),
            )
            self.B_im = self.param(
                "B_im",
                partial(matrix_init, normalization=jnp.sqrt(self.d_in)),
                (2*self.d_hidden, self.d_in),
            )
        else:
            self.B_re = self.param(
                "B_re",
                partial(matrix_init, normalization=jnp.sqrt(self.d_model)),
                (2*self.d_hidden, self.d_model),
            )
            self.B_im = self.param(
                "B_im",
                partial(matrix_init, normalization=jnp.sqrt(self.d_model)),
                (2*self.d_hidden, self.d_model),
            )

        if self.d_out is not None:
            self.C_re = self.param(
                "C_re",
                partial(matrix_init, normalization=jnp.sqrt(2 * self.d_hidden)),
                (self.d_out, 2*self.d_hidden),
            )
            self.C_im = self.param(
                "C_im",
                partial(matrix_init, normalization=jnp.sqrt(2 * self.d_hidden)),
                (self.d_out, 2*self.d_hidden),
            )
            self.D = self.param("D", matrix_init, (self.d_out,))
        elif self.has_layer_output:
            self.C_re = self.param(
                "C_re",
                partial(matrix_init, normalization=jnp.sqrt(2 * self.d_hidden)),
                (self.d_model, 2*self.d_hidden),
            )
            self.C_im = self.param(
                "C_im",
                partial(matrix_init, normalization=jnp.sqrt(2 * self.d_hidden)),
                (self.d_model, 2*self.d_hidden),
            )
            self.D = self.param("D", matrix_init, (self.d_model,))

        # Internal variables of the model needed for updating the gradient
        if self.online and self.approximation_type not in ["spatial", "reservoir"]:
            self.pert_hidden_states = self.variable(
                "perturbations",
                "hidden_states",
                partial(jnp.zeros, dtype=jnp.complex64),
                (2* self.d_hidden,),
            )

        if self.mixing == "full":
            self.normalizer = 0.5
        elif self.mixing == "symmetric":
            self.normalizer = 1.0
        elif self.mixing == "none":
            self.normalizer = 1.0
        elif self.mixing == "rotational":
            self.normalizer = 1.0
            self.phi = self.param("phi", matrix_init, (self.d_hidden,))
        else:
            raise ValueError("Mixing type not recognized")

    def __call__(self, inputs, raw_old_hidden_states, raw_traces=None):
        """
        Forward pass. If in training mode, additionally computes the eligibility traces that
        will be needed to compute the gradient estimate in backward.
        """

        old_hidden_states = jnp.reshape(raw_old_hidden_states, 2 * self.d_hidden)

        if self.mixing == "rotational":
            # Rotate old hidden states to the eigen basis. From here on, the variable old_hidden_states
            # is always in the basis in which the recurrence happens - used in both the recurrence and the trace updates
            old_hidden_states = self._to_eigen_basis(old_hidden_states)

        # Compute hidden states and outputs
        #hidden_states = self.get_hidden_states(inputs)
        raw_hidden_states = self.get_single_step_hidden_states(inputs, old_hidden_states)

        if self.has_non_linearity_in_recurrence:
            pre_activation_hidden_states = jnp.reshape(raw_hidden_states, (2, self.d_hidden))
            raw_hidden_states = jnp.tanh(raw_hidden_states)

        if self.online and self.approximation_type not in ["spatial", "reservoir"]:
            # To obtain the spatially backpropagated errors sent to hidden_states
            # NOTE: only works if pert_hidden_states is equal to 0
            raw_hidden_states += self.pert_hidden_states.value

        if self.mixing == "rotational":
            # Rotate back to the original basis, but only after injecting perturbations! Here, the gradients of the perturbations will now represent dL/dz instead of dL/dh.
            # For gradient computation, we will do dz/dtheta * dL/dz instead of dh/dtheta * dL/dh, so that we can keep the element-wise mult
            # From here on, the variable raw_hidden_states is always in the original basis - used for output computation and return
            raw_hidden_states = self._from_eigen_basis(raw_hidden_states)

        if self.has_layer_output:
            output = self.to_output_single_step(inputs, raw_hidden_states)
        else:
            output = raw_hidden_states

        hidden_states = jnp.reshape(raw_hidden_states, (2, self.d_hidden))

        if raw_traces is None:
            return output, hidden_states

        # Compute and update traces if needed (i.e. if we are in online training mode)
        if self.online and self.approximation_type not in ["spatial", "reservoir"]:

            # Unpack traces
            raw_lambda_trace, raw_gamma_trace, raw_B_trace = raw_traces
            traces = (
                jnp.reshape(raw_lambda_trace, (2, 4, self.d_hidden,)),
                jnp.reshape(raw_gamma_trace, (2, 2, self.d_hidden,)),
                jnp.reshape(raw_B_trace, (2, 2, self.d_hidden, self.d_model)) if self.d_in is None else jnp.reshape(raw_B_trace, (2, 2, self.d_hidden, self.d_in))
            )

            if self.mixing == "rotational":
                # Rotate traces to the eigen basis. Trace updates are always done in the eigen basis to keep the diagonal properties
                lambda_tr_h, gamma_tr_h, B_tr_h = traces

                # Bring H to axis 1 -> (2, H, *)
                lambda_tr_h_for_rot = jnp.swapaxes(lambda_tr_h, 1, 2)         # (2, H, 4)       
                gamma_tr_h_for_rot = jnp.swapaxes(gamma_tr_h, 1, 2)           # (2, H, 2)
                B_tr_h_for_rot = jnp.swapaxes(B_tr_h, 1, 2)                   # (2, H, 2, d_model) or (2, H, 2, d_in)

                # Rotate
                lambda_tr_z_for_rot = self._rotate_pair_to_eigen(lambda_tr_h_for_rot)  # (2, H, 4)
                gamma_tr_z_for_rot = self._rotate_pair_to_eigen(gamma_tr_h_for_rot)  # (2, H, 2)
                B_tr_z_for_rot = self._rotate_pair_to_eigen(B_tr_h_for_rot)  # (2, H, 2, d_model) or (2, H, 2, d_in)

                # Put it back into orignal shape
                lambda_tr_z = jnp.swapaxes(lambda_tr_z_for_rot, 1, 2)         # (2, 4, H)
                gamma_tr_z = jnp.swapaxes(gamma_tr_z_for_rot, 1, 2)           # (2, 2, H)
                B_tr_z = jnp.swapaxes(B_tr_z_for_rot, 1, 2)                   # (2, 2, H, d_model) or (2, 2, H, d_in)

                traces = (lambda_tr_z, gamma_tr_z, B_tr_z)

            Bu_elements = self.get_B() @ inputs

            if self.has_non_linearity_in_recurrence:
                # Create multiplier for chain rule derivative of tanh
                multiplier = (1 - hidden_states**2)
            else:
                # Identity multiplier
                multiplier = jnp.ones_like(hidden_states)

            # Update traces for B, lambda and gamma
            if self.approximation_type in ["1truncated"]:
                Lambda_elements = self.get_diag_lambda().reshape(4, self.d_hidden)

                # Update for trace lambda
                new_traces_lambda_src_node = jnp.zeros_like(traces[0][0])
                new_traces_lambda_src_node = new_traces_lambda_src_node.at[:2].add(self.normalizer * old_hidden_states.reshape(2, self.d_hidden))
                new_traces_lambda_src_node *= multiplier[0].reshape(1, self.d_hidden) # src hidden_states

                new_traces_lambda_dst_node = jnp.zeros_like(traces[0][1])
                new_traces_lambda_dst_node = new_traces_lambda_dst_node.at[2:].add(self.normalizer * old_hidden_states.reshape(2, self.d_hidden))
                new_traces_lambda_dst_node *= multiplier[1].reshape(1, self.d_hidden)

                # Update for trace gamma
                Bu_elements_gamma = Bu_elements.reshape(2, self.d_hidden) # (2*d_hidden)

                new_traces_gamma_src_node = jnp.zeros_like(traces[1][0])
                new_traces_gamma_src_node = new_traces_gamma_src_node.at[0].add(Bu_elements_gamma[0])
                new_traces_gamma_src_node *= multiplier[0].reshape(1, self.d_hidden)

                new_traces_gamma_dst_node = jnp.zeros_like(traces[1][1])
                new_traces_gamma_dst_node = new_traces_gamma_dst_node.at[1].add(Bu_elements_gamma[1])
                new_traces_gamma_dst_node *= multiplier[1].reshape(1, self.d_hidden)

            elif self.approximation_type in ["full", "full_rec", "full_rec_simpleB", "snap1"]:
                Lambda_elements = self.get_diag_lambda().reshape(4, self.d_hidden)

                # Update for trace lambda
                new_traces_lambda_src_node = (self.normalizer * Lambda_elements[0] * traces[0][0] + self.normalizer * Lambda_elements[1] * traces[0][1]) # lambda_elements[0] -> src_src, lambda_elements[1] -> src_dst
                new_traces_lambda_src_node = new_traces_lambda_src_node.at[:2].add(self.normalizer * old_hidden_states.reshape(2, self.d_hidden))
                new_traces_lambda_src_node *= multiplier[0].reshape(1, self.d_hidden) # src hidden_states

                new_traces_lambda_dst_node = self.normalizer *Lambda_elements[2] * traces[0][0] + self.normalizer *Lambda_elements[3] * traces[0][1] # lambda_elements[2] -> dst_src, lambda_elements[3] -> dst_dst
                new_traces_lambda_dst_node = new_traces_lambda_dst_node.at[2:].add(self.normalizer * old_hidden_states.reshape(2, self.d_hidden))
                new_traces_lambda_dst_node *= multiplier[1].reshape(1, self.d_hidden)
                
                # Update for trace gamma
                Bu_elements_gamma = Bu_elements.reshape(2, self.d_hidden) # (2*d_hidden)

                new_traces_gamma_src_node = self.normalizer * Lambda_elements[0] * traces[1][0] + self.normalizer * Lambda_elements[1] * traces[1][1] # lambda_elements[0] -> src_src, lambda_elements[1] -> src_dst
                new_traces_gamma_src_node = new_traces_gamma_src_node.at[0].add(Bu_elements_gamma[0])
                new_traces_gamma_src_node *= multiplier[0].reshape(1, self.d_hidden)

                new_traces_gamma_dst_node = self.normalizer * Lambda_elements[2] * traces[1][0] + self.normalizer * Lambda_elements[3] * traces[1][1] # lambda_elements[2] -> dst_src, lambda_elements[3] -> dst_dst
                new_traces_gamma_dst_node = new_traces_gamma_dst_node.at[1].add(Bu_elements_gamma[1])
                new_traces_gamma_dst_node *= multiplier[1].reshape(1, self.d_hidden)

            # Update trace for B
            if self.approximation_type in ["full", "snap1"]:
                full_Lambda_elements = self.get_diag_lambda().reshape(4, self.d_hidden, 1)

                gammau_elements = jnp.outer(self.get_diag_gamma(), inputs
                ).astype(jnp.complex64).reshape(2, self.d_hidden, self.d_model) if self.d_in is None else jnp.outer(self.get_diag_gamma(), inputs).astype(jnp.complex64).reshape(2, self.d_hidden, self.d_in) # (2, d_hidden, d_model) or (2, d_hidden, d_in)

                new_traces_B_src_node = self.normalizer * full_Lambda_elements[0] * traces[2][0] + self.normalizer * full_Lambda_elements[1] * traces[2][1] # lambda_elements[0] -> src_src, lambda_elements[1] -> src_dst
                new_traces_B_src_node = new_traces_B_src_node.at[0].add(gammau_elements[0])
                new_traces_B_src_node *= multiplier[0].reshape(1, self.d_hidden, 1)

                new_traces_B_dst_node = self.normalizer * full_Lambda_elements[2] * traces[2][0] + self.normalizer * full_Lambda_elements[3] * traces[2][1] # lambda_elements[2] -> dst_src, lambda_elements[3] -> dst_dst
                new_traces_B_dst_node = new_traces_B_dst_node.at[1].add(gammau_elements[1])
                new_traces_B_dst_node *= multiplier[1].reshape(1, self.d_hidden, 1)
            else:
                new_traces_B_src_node = jnp.zeros_like(traces[2][0])
                new_traces_B_dst_node = jnp.zeros_like(traces[2][1])
                
        new_traces = [
            jnp.stack(
                [new_traces_lambda_src_node, new_traces_lambda_dst_node], axis=0
            ).reshape(2, -1),
            jnp.stack(
                [new_traces_gamma_src_node, new_traces_gamma_dst_node], axis=0
            ).reshape(2, -1),
            jnp.stack(
                [new_traces_B_src_node, new_traces_B_dst_node], axis=0
            ).reshape(2, -1, self.d_model) if self.d_in is None else jnp.stack(
                [new_traces_B_src_node, new_traces_B_dst_node], axis=0
            ).reshape(2, -1, self.d_in)
        ]
        if self.mixing == "rotational":
            # Rotate traces back to the original basis
            new_lambda_tr_z, new_gamma_tr_z, new_B_tr_z = new_traces

            # Bring H to axis 1 -> (2, H, *)
            new_lambda_tr_z_for_rot_back = jnp.reshape(new_lambda_tr_z, (2, 4, self.d_hidden))      # (2, 4, H)
            new_lambda_tr_z_for_rot_back = jnp.swapaxes(new_lambda_tr_z_for_rot_back, 1, 2)         # (2, H, 4)

            new_gamma_tr_z_for_rot_back = jnp.reshape(new_gamma_tr_z, (2, 2, self.d_hidden))        # (2, 2, H)
            new_gamma_tr_z_for_rot_back = jnp.swapaxes(new_gamma_tr_z_for_rot_back, 1, 2)           # (2, H, 2)

            new_B_tr_z_for_rot_back = jnp.reshape(new_B_tr_z, (2, 2, self.d_hidden, -1))            # (2, 2, H, d_model) or (2, 2, H, d_in)
            new_B_tr_z_for_rot_back = jnp.swapaxes(new_B_tr_z_for_rot_back, 1, 2)                   # (2, H, 2, d_model) or (2, H, 2, d_in)

            # Rotate back
            new_lambda_tr_h_for_rot = self._rotate_pair_from_eigen(new_lambda_tr_z_for_rot_back)    # (2, H, 4)
            new_gamma_tr_h_for_rot = self._rotate_pair_from_eigen(new_gamma_tr_z_for_rot_back)      # (2, H, 2)
            new_B_tr_h_for_rot = self._rotate_pair_from_eigen(new_B_tr_z_for_rot_back)         # (2, H, 2, d_model) or (2, H, 2, d_in)

            # Put it back into orignal shape
            new_lambda_tr_h = jnp.swapaxes(new_lambda_tr_h_for_rot, 1, 2)         # (2, 4, H)
            new_gamma_tr_h = jnp.swapaxes(new_gamma_tr_h_for_rot, 1, 2)           # (2, 2, H)
            new_B_tr_h = jnp.swapaxes(new_B_tr_h_for_rot, 1, 2)                   # (2, 2, H, d_model) or (2, 2, H, d_in)

            # Flatten to return same shape as self.mixing != 'rotational'
            new_lambda_tr_flat = jnp.reshape(new_lambda_tr_h, (2, -1))
            new_gamma_tr_flat =  jnp.reshape(new_gamma_tr_h, (2, -1))
            new_B_tr_flat = jnp.reshape(new_B_tr_h, (2, -1, self.d_model)) if self.d_in is None else jnp.reshape(new_B_tr_h, (2, -1, self.d_in))

            new_traces = [
                new_lambda_tr_flat,
                new_gamma_tr_flat,
                new_B_tr_flat
            ]

        return output, hidden_states, tuple(new_traces)

    def update_gradients(self, grad, raw_traces, perturbation_grads):
        """
        Eventually combine traces and perturbations to compute the (online) gradient.
        """
        if self.training_mode in ["bptt", "online_spatial", "online_reservoir"]:
            raise ValueError("Upgrade gradient should not be called for this training mode")

        # We need to change the gradients for lambda, gamma and B
        # The others are automatically computed with spatial backpropagation

        # Unpack traces
        raw_lambda_trace, raw_gamma_trace, raw_B_trace = raw_traces
        traces = (
                jnp.reshape(raw_lambda_trace, (2, 4, self.d_hidden,)),
                jnp.reshape(raw_gamma_trace, (2, 2, self.d_hidden,)),
                jnp.reshape(raw_B_trace, (2, 2, self.d_hidden, self.d_model)) if self.d_in is None else jnp.reshape(raw_B_trace, (2, 2, self.d_hidden, self.d_in))
            )

        dL = perturbation_grads['hidden_states'].reshape(2, 1, self.d_hidden)

        if self.mixing == "rotational":
            # ---- Convert traces from h-basis to z-basis ----
            lambda_tr_h, gamma_tr_h, B_tr_h = traces
            # lambda traces: (2, 4, H) -> (2, H, 4) -> rotate -> (2, H, 4) -> (2, 4, H)
            lambda_tr_h_for_rot = jnp.swapaxes(lambda_tr_h, 1, 2)              # (2, H, 4)
            lambda_tr_z_for_rot = self._rotate_pair_to_eigen(lambda_tr_h_for_rot)  # (2, H, 4)
            lambda_tr_z = jnp.swapaxes(lambda_tr_z_for_rot, 1, 2)              # (2, 4, H)

            # gamma traces: (2, 2, H) -> (2, H, 2) -> rotate -> (2, H, 2) -> (2, 2, H)
            gamma_tr_h_for_rot = jnp.swapaxes(gamma_tr_h, 1, 2)               # (2, H, 2)
            gamma_tr_z_for_rot = self._rotate_pair_to_eigen(gamma_tr_h_for_rot)    # (2, H, 2)
            gamma_tr_z = jnp.swapaxes(gamma_tr_z_for_rot, 1, 2)               # (2, 2, H)

            # B traces: (2, 2, H, D) -> (2, H, 2, D) -> rotate -> (2, H, 2, D) -> (2, 2, H, D)
            B_tr_h_for_rot = jnp.swapaxes(B_tr_h, 1, 2)                       # (2, H, 2, D)
            B_tr_z_for_rot = self._rotate_pair_to_eigen(B_tr_h_for_rot)       # (2, H, 2, D)
            B_tr_z = jnp.swapaxes(B_tr_z_for_rot, 1, 2)                        # (2, 2, H, D)

            traces = (lambda_tr_z, gamma_tr_z, B_tr_z)

        delta_lambda = jnp.sum(dL * traces[0], axis=0).reshape(-1)

        _, dl = jax.vjp(
            lambda nu, theta: self.get_diag_lambda(nu=nu, theta=theta),
            self.nu,
            self.theta,
        )
        grad_nu, grad_theta = dl(delta_lambda)
        grad["nu"] = grad_nu
        grad["theta"] = grad_theta

        # Grads for gamma if needed
        if self.gamma_norm:

            delta_gamma = jnp.sum(dL * traces[1], axis=0).reshape(-1).real

            # as dgamma/dgamma_log = exp(gamma_log) = gamma
            grad["gamma_log"] = delta_gamma * self.get_diag_gamma()

        # Grads for B
        if self.approximation_type in ["snap1", "full", "full_rec_simpleB"]:

            grad_B = jnp.sum(dL.reshape(2, 1, self.d_hidden, 1) * traces[2], axis=0).reshape(-1, self.d_model) if self.d_in is None else jnp.sum(dL.reshape(2, 1, self.d_hidden, 1) * traces[2], axis=0).reshape(-1, self.d_in) # (2*d_hidden, d_model) or (2*d_hidden, d_in)

            grad["B_re"] = grad_B.real
            grad["B_im"] = -grad_B.imag  # Comes from the use of Writtinger derivatives

        return grad