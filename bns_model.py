import tensorflow as tf
import numpy as np

class BayesianDense(tf.keras.layers.Layer):
    """Mean-field Gaussian variational layer with trainable σ_w."""

    def __init__(self, units, kl_weight=1e-5, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units      = units
        self.kl_weight  = kl_weight
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.w_mu  = self.add_weight(name="w_mu",  shape=(d, self.units),
                                     initializer="glorot_uniform")
        self.w_rho = self.add_weight(name="w_rho", shape=(d, self.units),
                                     initializer=tf.constant_initializer(-4.0))
        self.b_mu  = self.add_weight(name="b_mu",  shape=(self.units,),
                                     initializer="zeros")
        self.b_rho = self.add_weight(name="b_rho", shape=(self.units,),
                                     initializer=tf.constant_initializer(-4.0))

    def call(self, inputs, training=False):
        w_sigma = tf.nn.softplus(self.w_rho) + 1e-6
        b_sigma = tf.nn.softplus(self.b_rho) + 1e-6

        if training:
            w = self.w_mu + w_sigma * tf.random.normal(tf.shape(self.w_mu))
            b = self.b_mu + b_sigma * tf.random.normal(tf.shape(self.b_mu))
        else:
            w = self.w_mu
            b = self.b_mu

        if training:
            kl = 0.5 * tf.reduce_sum(
                self.w_mu**2 + w_sigma**2 - tf.math.log(w_sigma**2) - 1.0
            )
            kl += 0.5 * tf.reduce_sum(
                self.b_mu**2 + b_sigma**2 - tf.math.log(b_sigma**2) - 1.0
            )
            self.add_loss(self.kl_weight * kl)

        out = tf.matmul(inputs, w) + b
        return self.activation(out) if self.activation else out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(units=self.units, kl_weight=self.kl_weight,
                   activation=tf.keras.activations.serialize(self.activation))
        return cfg

def build_bnn(input_dim, num_classes, hidden_sizes, kl_weight, name="BNN"):
    inp = tf.keras.Input(shape=(input_dim,), name="input")
    x   = inp
    for i, h in enumerate(hidden_sizes):
        x = BayesianDense(h, kl_weight=kl_weight,
                          activation="relu", name=f"bayes_{i}")(x)
    out = BayesianDense(num_classes, kl_weight=kl_weight, name="logits")(x)
    return tf.keras.Model(inp, out, name=name)

def compute_empirical_mu(X_raw_hours, y, high_class=2):
    mu = np.array([
        np.mean(y[X_raw_hours == h] == high_class) if np.any(X_raw_hours == h) else 0.0
        for h in range(24)
    ], dtype=np.float32)
    return mu

class BNS_Model(tf.keras.Model):
    def __init__(self, backbone, empirical_mu, N_train,
                 lambda_max=0.05, beta=1.0, high_class=2):
        super().__init__()
        self.backbone     = backbone
        self.emp_mu       = tf.constant(empirical_mu, dtype=tf.float32)
        self.N_train      = N_train
        self.lambda_max   = lambda_max
        self.lambda_t     = lambda_max  # Static for streamlit demo
        self.beta         = beta
        self.high_class   = high_class

        self._loss_tr = tf.keras.metrics.Mean("loss")
        self._acc_tr  = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
        self._tsc_tr  = tf.keras.metrics.Mean("tsc")

    @property
    def metrics(self):
        return [self._loss_tr, self._acc_tr, self._tsc_tr]

    def call(self, inputs, training=False):
        return self.backbone(inputs, training=training)

    def _compute_tsc(self, logits, hours):
        probs      = tf.nn.softmax(logits)
        high_probs = probs[:, self.high_class]
        hours_i32  = tf.cast(hours, tf.int32)

        mu_hat = []
        for h in range(24):
            mask  = tf.cast(tf.equal(hours_i32, h), tf.float32)
            num   = tf.reduce_sum(mask * high_probs)
            den   = tf.reduce_sum(mask) + 1e-6
            mu_hat.append(num / den)

        mu_hat = tf.stack(mu_hat)
        tsc    = tf.reduce_mean(tf.square(mu_hat - self.emp_mu))
        return tsc

    def train_step(self, data):
        x_full, y = data
        x_feat = x_full[:, :-1]
        hours  = x_full[:, -1]

        with tf.GradientTape() as tape:
            logits = self.backbone(x_feat, training=True)
            ce = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y, logits, from_logits=True))

            kl_losses = self.backbone.losses
            kl        = tf.add_n(kl_losses) if kl_losses else 0.0
            kl_scaled = self.beta * kl / self.N_train

            tsc   = self._compute_tsc(logits, hours)
            loss  = ce + kl_scaled + self.lambda_t * tsc

        grads = tape.gradient(loss, self.backbone.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.backbone.trainable_variables))

        self._loss_tr.update_state(loss)
        self._acc_tr.update_state(y, logits)
        self._tsc_tr.update_state(tsc)

        return {m.name: m.result() for m in self.metrics}
